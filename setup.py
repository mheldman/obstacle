#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add comments
"""

import io
import os
import sys
import subprocess
from shutil import rmtree

import setuptools
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from setuptools import find_packages, setup, Command

version = '0.0.1'
isreleased = True

install_requires = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'pytest>=2',
    'pybind11>=2.2'
)


# set the version information
# https://github.com/numpy/numpy/commits/master/setup.py
# Return the git revision as a string


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        GIT_BRANCH = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'
        GIT_BRANCH = ''

    return GIT_REVISION


def set_version_info(VERSION, ISRELEASED):
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('obstacle/version.py'):
        try:
            import imp
            version = imp.load_source("obstacle.version", "obstacle/version.py")
            GIT_REVISION = version.git_revision
        except ImportError:
            raise ImportError('Unable to read version information.')
    else:
        GIT_REVISION = 'Unknown'
        GIT_BRANCH = ''

    FULLVERSION = VERSION
    if not ISRELEASED:
        FULLVERSION += '.dev0' + '+' + GIT_REVISION[:7]

    print(GIT_REVISION)
    print(FULLVERSION)
    return FULLVERSION, GIT_REVISION


def write_version_py(VERSION,
                     FULLVERSION,
                     GIT_REVISION,
                     ISRELEASED,
                     filename='obstacle/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


fullversion, git_revision = set_version_info(version, isreleased)
write_version_py(version, fullversion, git_revision, isreleased,
                 filename='obstacle/version.py')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        pytest.main(self.test_args)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, 'std=c++17'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

    # identify extension modules
    # since numpy is needed (for the path), need to bootstrap the setup
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

pfas_core_headers = ['relaxation.h', 'monotone_restriction.h']
pfas_core_headers = [f.replace('.h', '') for f in pfas_core_headers]

ext_modules = [Extension('obstacle.pfas_core.%s' % f,
                         sources=['obstacle/pfas_core/%s_bind.cpp' % f],
                         include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
                         language='c++') for f in pfas_core_headers]


class UploadCommand(Command):
  """Support setup.py upload."""
    
  description = 'Build and publish the package.'
  user_options = []
  
  @staticmethod
  def status(s):
    """Prints things in bold."""
    print('\033[1m{0}\033[0m'.format(s))
  
  def initialize_options(self):
    pass
  
  def finalize_options(self):
    pass
  
  def run(self):
    try:
      self.status('Removing previous builds…')
      rmtree(os.path.join(here, 'dist'))
    except OSError:
      pass
    
    self.status('Building Source and Wheel (universal) distribution…')
    os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))
    
    self.status('Uploading the package to PyPI via Twine…')
    os.system('twine upload dist/*')
    
    self.status('Pushing git tags…')
    os.system('git tag v{0}'.format(about['__version__']))
    os.system('git push --tags')

    sys.exit()


setup(
    name='obstacle',
    version=fullversion,
    keywords=['elliptic pde poisson laplace obstacle problem linear complementarity problem variational inequality'],
    author='Max Heldman',
    author_email='mheldman1@gmail.com',
    maintainer='Max Heldman',
    maintainer_email='mheldman1@gmail.com',
    url='https://github.com/mheldman/obstacle',
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    description=__doc__.split('\n')[0],
    long_description=__doc__,
    #
    packages=find_packages(exclude=['doc']),
    include_package_data=False,
    install_requires=install_requires,
    zip_safe=False,
    #
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt, 'test': PyTest},
    setup_requires=['numpy', 'pybind11'],
    #
    tests_require=['pytest'],
    #
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
