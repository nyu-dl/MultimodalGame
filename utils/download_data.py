"""
Script to download images and create descriptions file.

Usage:

    wget http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz # download imagenet urls. please decompress.
    python --cmd_urls # get relevant urls from imagenet
    python --cmd_split # create train/dev/test splits of urls
    python --cmd_desc # create descriptions file
    python --cmd_download # download files for each split/class

Some sample synsets from Imagenet:

    n01498041 stingray
    n01514859 hen
    n01518878 ostrich
    n01531178 goldfinch
    n01558993 robin
    n01580077 jay
    n01582220 magpie
    n01592084 chickadee
    n01616318 vulture
    n01641577 bullfrog
    n01667778 terrapin
    n01687978 agama
    n01704323 triceratops
    n01768244 trilobite
    n01770393 scorpion
    n01774750 tarantula
    n01784675 centipede
    n01806143 peacock
    n01806567 quail
    n01807496 partridge
    n01818515 macaw
    n01820546 lorikeet
    n01833805 hummingbird
    n01843065 jacamar
    n01847000 drake
    n01855672 goose
    n01910747 jellyfish
    n01944390 snail
    n01945685 slug
    n01882714 koala

"""

from __future__ import print_function

from collections import OrderedDict
import os
import sys
import json
import time
import random
import urllib
import threading
from tqdm import tqdm
from parse import *

from nltk.corpus import wordnet as wn

import gflags

FLAGS = gflags.FLAGS


def try_mkdir(path):
    try:
        os.mkdir(path)
        return 1
    except BaseException as e:
        # directory already exists
        return 0


def flickr_name(url):
    tpl = "http://{subdomain}.flickr.com/{part1}/{part2}.{suffix}"
    data = parse(tpl, url)
    return "{subdomain}_{part1}_{part2}.{suffix}".format(**data.named)


class MultiThreadedDownloader(object):

    def __init__(self, download_path, num_threads, urls, time_wait):
        self.lock = threading.Lock()
        self.download_path = download_path
        self.num_threads = num_threads
        self.urls = urls
        self.index = 0
        self.time_wait = time_wait
        self.pbar = tqdm(total=len(self.urls))

    def worker(self):
        finished = False
        while True:
            self.lock.acquire()
            try:
                if self.index < len(self.urls):
                    # atomically acquire index
                    url = self.urls[self.index]
                    _filename = flickr_name(url)
                    _save_path = os.path.join(self.download_path, _filename)

                    # increment index
                    self.index = self.index + 1
                    self.pbar.update(1)
                else:
                    finished = True
            finally:
                self.lock.release()

            # if no urls left, break loop
            if finished:
                break

            # download url
            if not os.path.exists(_save_path):
                urllib.urlretrieve(url, _save_path)
                saved = True
                time.sleep(self.time_wait)

    def run(self):
        # start threads
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker, args=())
            t.start()
            threads.append(t)
            time.sleep(self.time_wait)

        # wait until all threads complete
        for t in threads:
            t.join()

        self.pbar.close()


def cmd_urls():

    random.seed(FLAGS.seed)

    assert os.path.exists(FLAGS.save_urls_path), "Make sure to create urls directory: {}".format(FLAGS.save_urls_path)

    synsets = FLAGS.synsets.split(',')
    classes = FLAGS.classes.split(',')
    synsets_to_class = {ss: cc for ss, cc in zip(synsets, classes)}
    urls = OrderedDict()
    for k in classes:
        urls[k] = []

    # read urls
    with open(FLAGS.load_imagenet_path) as f:
        for ii, line in enumerate(f):
            try:
                line = line.strip()
                _synset, _url = line.split('\t')
                _synset = _synset.split('_')[0]
                if _synset in synsets and FLAGS.filter_url in _url:
                    _class = synsets_to_class[_synset]
                    urls[_class].append(_url)
            except:
                print("skipping line {}: {}".format(ii, line))

    # randomize and restrict to limit
    for k in urls.keys():
        random.shuffle(urls[k])
        urls[k] = urls[k][:FLAGS.class_size]
        assert len(urls[k]) == FLAGS.class_size, "Not enough urls for: {} ({})".format(k, len(urls[k]))

    # write to file
    for k in urls.keys():
        with open("{}/{}.txt".format(FLAGS.save_urls_path, k), "w") as f:
            for _url in urls[k]:
                f.write(_url + '\n')


def cmd_split():

    random.seed(FLAGS.seed)

    datasets = dict(train=dict(), dev=dict(), test=dict())

    for cls in FLAGS.classes.split(','):

        with open("{}/{}.txt".format(FLAGS.load_urls_path, cls)) as f:
            urls = [line.strip() for line in f]

        assert len(urls) >= FLAGS.train_size + FLAGS.dev_size + FLAGS.test_size, \
            "There are not sufficient urls for class: {}".format(cls)

        random.shuffle(urls)

        # Train
        offset = 0
        size = FLAGS.train_size
        datasets['train'][cls] = urls[offset:offset + size]

        # Dev
        offset += FLAGS.train_size
        size = FLAGS.dev_size
        datasets['dev'][cls] = urls[offset:offset + size]

        # Test
        offset += FLAGS.dev_size
        size = FLAGS.test_size
        datasets['test'][cls] = urls[offset:offset + size]

    with open(FLAGS.save_datasets_path, "w") as f:
        f.write(json.dumps(datasets, indent=4, sort_keys=True))


def cmd_desc():
    animal = wn.synset('animal.n.01')

    descriptions = OrderedDict()

    # get animal synset for each class, and the class's wordnet description
    for cls in FLAGS.classes.split(','):
        for i in range(1, 10):
            _synset = wn.synset('{}.n.0{}'.format(cls, i))
            if _synset.lowest_common_hypernyms(animal)[0] == animal:
                break

        if _synset.lowest_common_hypernyms(animal)[0] != animal:
            raise BaseException("No animal synset found for: {}".format(cls))

        descriptions[cls] = _synset.definition()

    # write to descriptions file
    with open(FLAGS.save_descriptions_path, "w") as f:
        for ii, cls in enumerate(sorted(descriptions.keys())):
            desc = descriptions[cls].replace(',', '')
            f.write("{},{},{}\n".format(ii, cls, desc))


def cmd_download():

    with open(FLAGS.load_datasets_path) as f:
        datasets = json.loads(f.read())

    for _d in ['train', 'dev', 'test']:
        _dataset_path = os.path.join(FLAGS.save_images, _d)
        try_mkdir(_dataset_path)

        for cls in FLAGS.classes.split(','):
            _dataset_cls_path = os.path.join(_dataset_path, cls)
            try_mkdir(_dataset_cls_path)

            print("Downloading images for {}/{}".format(_d, cls))

            urls = datasets[_d][cls]
            downloader = MultiThreadedDownloader(_dataset_cls_path, FLAGS.num_threads, urls, FLAGS.throttle)
            downloader.run()


if __name__ == '__main__':
    gflags.DEFINE_string("synsets", "n01498041,n01514859,n01518878,n01531178,n01558993,n01580077" \
        ",n01582220,n01592084,n01616318,n01641577,n01667778,n01687978,n01704323,n01768244,n01770393" \
        ",n01774750,n01784675,n01806143,n01806567,n01807496,n01818515,n01820546,n01833805,n01843065" \
        ",n01847000,n01855672,n01910747,n01944390,n01945685,n01882714", "Comma-delimited list of sysnet ids to use.")
    gflags.DEFINE_string("classes", "stingray,hen,ostrich,goldfinch,robin,jay,magpie" \
        ",chickadee,vulture,bullfrog,terrapin,agama,triceratops,trilobite,scorpion,tarantula" \
        ",centipede,peacock,quail,partridge,macaw,lorikeet,hummingbird,jacamar,drake,goose" \
        ",jellyfish,snail,slug,koala", "Comma-delimited list of classes to use. Should match sysnet ids.")
    gflags.DEFINE_integer("seed", 11, "Seed for shuffling urls.")

    # urls args
    gflags.DEFINE_string("load_imagenet_path", "./fall11_urls.txt", "Path to imagenet urls.")
    gflags.DEFINE_string("save_urls_path", "./urls", "Path to directory with url files.")
    gflags.DEFINE_integer("class_size", 500, "Size of urls to keep (in images per class).")
    gflags.DEFINE_string("filter_url", "static.flickr", "String to filter urls.")

    # split args
    gflags.DEFINE_string("load_urls_path", "./urls", "Path to directory with url files.")
    gflags.DEFINE_string("save_datasets_path", "datasets.json", "Single JSON file defining train/dev/test splits.")
    gflags.DEFINE_integer("train_size", 100, "Size of dataset (in images per class).")
    gflags.DEFINE_integer("dev_size", 100, "Size of dataset (in images per class).")
    gflags.DEFINE_integer("test_size", 100, "Size of dataset (in images per class).")

    # description args
    gflags.DEFINE_string("load_datasets_path", "datasets.json", "Single JSON file defining train/dev/test splits.")
    gflags.DEFINE_string("save_images", "./imgs", "Path to save images.")

    # download args
    gflags.DEFINE_string("save_descriptions_path", "./descriptions.csv", "Path to descriptions file.")
    gflags.DEFINE_integer("num_threads", 8, "Use a multi-threaded image downloader.")
    gflags.DEFINE_integer("throttle", 0.01, "Throttle the downloader.")

    # commands
    gflags.DEFINE_boolean("cmd_urls", False, "Extract relevant urls from imagenet.")
    gflags.DEFINE_boolean("cmd_split", False, "Split urls into datasets.")
    gflags.DEFINE_boolean("cmd_desc", False, "Create descriptions file.")
    gflags.DEFINE_boolean("cmd_download", False, "Download images from flickr.")

    FLAGS(sys.argv)

    print("Flag Values:\n" + json.dumps(FLAGS.flag_values_dict(), indent=4, sort_keys=True))

    if FLAGS.cmd_urls:
        cmd_urls()
    if FLAGS.cmd_split:
        cmd_split()
    if FLAGS.cmd_desc:
        cmd_desc()
    if FLAGS.cmd_download:
        cmd_download()
