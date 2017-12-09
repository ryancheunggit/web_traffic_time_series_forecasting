# crawling data from 2017-09-01 to 2017-09-07
# crap code, does not seperate out elements out correctly.
import urllib
import pandas as pd
import numpy as np
import multiprocessing
import warnings
from tqdm import tqdm
tqdm.pandas(desc="progress monit")
import json
warnings.filterwarnings("ignore")

def parse_page_components(page):
    '''
        name, project, access, agent = parse_page_components(page)
    '''
    language = page.split(".wikipedia.org")[0].split("_")[-1]
    project = language + '.wikipedia.org'
    name = page[:page.find(project)][:-1]
    name = urllib.parse.quote_plus(name)
    access = page.split(".org")[-1][1:].split("_")[0]
    agent = page.split(".org")[-1][1:].split("_")[1]
    return name, project, access, agent

def get_views(page):
    '''
        the url format
        url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{0}/{1}/{2}/{3}/daily/{4}/{5}'.format(
            project, access, agent, name, '20170830', '20170910'
        )
    '''
    #print(series)
    name, project, access, agent = parse_page_components(page)
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{0}/{1}/{2}/{3}/daily/{4}/{5}'.format(
            project, access, agent, name, '20170911', '20170912'
        )
    views = 0
    try:
        url = urllib.request.urlopen(url)
        api_res = json.loads(url.read().decode())['items']
        views =  api_res[0]['views']
    except:
        pass
    return views


def main():
    train = pd.read_csv("../input/train_2.csv")
    train['2017-09-11'] = train.Page.progress_apply(get_views)
    train.to_csv("../input/train_3.csv", index = False)

if __name__ == '__main__':
    main()
