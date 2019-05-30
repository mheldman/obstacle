"""obstacle problem"""
from __future__ import absolute_import

from .obstaclep import *
from .Poisson2D import *

__all__ = [s for s in dir() if not s.startswith('_')]
