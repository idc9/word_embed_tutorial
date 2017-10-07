
import requests
import ast
import json

username = 'unc_networks'
password = 'UNCSTATS'


def case_info(op_id):
    """
    Given the case id returns a link to the opinion file on court listener
    """
    # API url
    op_api = 'https://www.courtlistener.com/api/rest/v3/opinions/%s/?format=json'\
              % op_id

    # get data from opinion and cluster api
    op_data = url_to_dict(op_api)
    cl_data = url_to_dict(op_data['cluster'])

    # url to opinion text
    opinion_url = 'https://www.courtlistener.com' + op_data['absolute_url']

    print cl_data['case_name']
    print 'date_filed: %s' % cl_data['date_filed']
    print opinion_url


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

