from setuptools import setup, find_packages

setup(
    name="HKMTR_LPR",
    version="0.1",
    author="chen xiaodong",
    author_email="xcheneo@connect.ust.hk",
    description="This is the initial version for LPR project. It only supports entry/exit one LP recognition.",

    # project homepage
    url="https://github.com/JRonRoad/hklpr_yolov5", 

    packages=find_packages(),

    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    package_data = {
        './': ['lpr_mtr.py', 'lpr_mtr.yaml'],
    },

    install_requires = [
        'Cython',
        'matplotlib>=3.2.2',
        'numpy>=1.18.5',
        'opencv-python>=4.1.2',
        'pillow',
        'PyYAML>=5.3',
        'scipy>=1.4.1',
        'tensorboard>=2.2',
        'torch==1.7.1',
        'torchvision==0.8.2',
        'tqdm>=4.41.0',
    ],
)