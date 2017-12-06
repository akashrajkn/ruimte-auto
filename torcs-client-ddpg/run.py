#! /usr/bin/env python3

import json

from pytocl.main import main
from my_driver import MyDriver

if __name__ == '__main__':
    with open('BAD', 'w+') as f:
        f.write('')

    main(MyDriver())
