import yaml

data = yaml.load(open('babyslakh_16k\Track00002\metadata.yaml'), Loader=yaml.Loader)

for stem in data['stems'].keys():
    print(data['stems'][stem]['inst_class'])