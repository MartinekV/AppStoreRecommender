from env.langdetect.detector_factory import DetectorFactory
import sqlite3
from env.langdetect.lang_detect_exception import LangDetectException


factory = DetectorFactory()
factory.load_profile('../env/langdetect/profiles')


def detect(text):
    detector = factory.create()
    detector.append(text)
    return detector.detect()


def detect_langs(text):
    detector = factory.create()
    detector.append(text)
    return detector.get_probabilities()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


print(detect("english words"))


def readSqliteTable():
    try:
        sqliteConnection = sqlite3.connect('../appleStore.sqlite3')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select_query = """SELECT * from apps_app"""
        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()
        print("Total rows are:  ", len(records))

        # find?
        if True:
            # once it is working invert the search and collect nonEnglish
            nonEnglishIDs = []
            descriptionIndex = 17
            nameIndex = 2
            seachBy = nameIndex
            for item in records:
                # item is not always correctly categorized by name, consider detecting language using description
                print(item[1])
                try:
                    if detect(item[seachBy]) != "en":
                        nonEnglishIDs.append(item)
                except LangDetectException:
                    # delete descriptions containing only numbers
                    if is_number(item[seachBy]):
                        nonEnglishIDs.append(item[1])
            print("non english rows are ", len(nonEnglishIDs))

        # delete?
        if False:
            # delte id 1061042798  1062209068  1064830114  1066737409   1069796800   1070702119   1073081246   1086006702   1087011731   1087200930   1087738560  1089810837
            # ToDo: detect chinese and japanise and so on and delete it
            sql_update_query = """DELETE from apps_app where id = ?"""
            for item in nonEnglishIDs:
                print(item[1])
                cursor.execute(sql_update_query, (item[1],))
            sqliteConnection.commit()
            print("Records deleted successfully")

        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")

readSqliteTable()
