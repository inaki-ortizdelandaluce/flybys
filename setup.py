import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

with open('README.md') as _f:
    _README_MD = _f.read()

_VERSION = '0.1'

setuptools.setup(
	name='flybys',
    version=_VERSION,
    description='A Python module to compute magnetopause and bowshock crossing events during BepiColombo flybys to Venus and Mercury',
    long_description=_README_MD,
    author='Inaki Ortiz de Landaluce',  
    author_email='inaki.ortizdelandaluce@gmail.com',
	packages=['flybys'],
    test_suite="tests",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
	install_requires=install_requires,
    license='MIT'
)