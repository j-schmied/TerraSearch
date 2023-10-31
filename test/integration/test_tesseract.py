import pytesseract


def main():
    try:
        version = pytesseract.get_tesseract_version()
        print("[+] Passed (Tesseract version: {})".format(version))
    except Exception as e:
        print("[-] Failed ({})".format(e))


if __name__ == '__main__':
    main()
