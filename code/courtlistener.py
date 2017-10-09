
import requests
import ast
import json


def url_to_dict(url):
    """
    :param url: String representing a json-style object on Court Listener's
    REST API

    :return: html_as_dict, a dictionary of the data on the HTML page
    """
    response = requests.get(url, auth=(username, password))
    html = response.text
    html = html.replace('false', 'False')
    html = html.replace('true', 'True')
    html = html.replace('null', 'None')
    html_as_dict = ast.literal_eval(html)
    return html_as_dict


def json_to_dict(path):
    """
    Given a path to a .json file returns a dict of the .json file

    Parameters
    ----------

    path: path to .json file

    Output:
    dict of json file
    """
    with open(path) as data_file:
        data = json.load(data_file)
        return data

