#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 M2KT Contributors. Licensed under the MIT License.

import logging


def set_logging(logfile='expe_log.log'):
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s-%(levelname)s:%(message)s',
        datefmt='%m-%d %H:%M:%S')

    # FileHandler for output to file
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # StreamHandler for output to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
