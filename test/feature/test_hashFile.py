from hashlib import md5
from sys import argv


def main():
    if len(argv) != 2:
        exit(0)

    document_path = argv[1]

    hasher = md5()
    bs = 65536

    with open(document_path, 'rb') as fp:
        while True:
            data = fp.read(bs)
            if not data:
                break
            hasher.update(data)

    print(hasher.hexdigest())


if __name__ == '__main__':
    main()
