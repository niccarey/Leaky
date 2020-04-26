#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import speed_tests

cProfile.runctx("speed_tests.nav_func_test('./testspeed_im.png', './test2.png')", globals(), locals(), "Profile.prof")

s=pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
