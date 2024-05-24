import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """
    Loads two csv-files and merges them together. Applies some cleaning to the categories-table.
    :param messages_filepath: message-date in csv-format
    :param categories_filepath: categories-data in csv-format
    :return: pandas df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda col: col[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda row: row[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicates from input df
    :param df: input df
    :return: pandas df
    """
    duplicates_mask = df.duplicated()
    print(f"{df[duplicates_mask].shape[0]} duplicates will be removed.")
    df = df.drop_duplicates()
    assert df.duplicated().sum() == 0, "There are still duplicate rows in the df!"

    return df


def save_data(df, database_filename):
    """
    Save the processed df to the db
    :param df: processed df
    :param database_filename: DB to save to
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages_categories', engine, index=False, if_exists='replace')

    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
