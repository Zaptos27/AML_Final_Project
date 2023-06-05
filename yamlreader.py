import yaml

def main():
    data = yaml.load(open('babyslakh_16k\Track00002\metadata.yaml'), Loader=yaml.Loader)

    for stem in data['stems'].keys():
        print(stem, ": ", data['stems'][stem]['inst_class'])

if __name__ == "__main__":
    main()