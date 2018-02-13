from shapeworld import dataset
import pprint

dataset = dataset(dtype='agreement', name='oneshape', )
generated = dataset.generate(n=3, mode='train', noise_range=0.1, include_model=True)
vocabulary = dataset.vocabularies['language']
pprint.pprint(vocabulary)
idx2word = {}
for k, v in vocabulary.items():
    idx2word[v] = k

# given to the image caption agreement system
batch = (generated['world'].shape, generated['caption'], generated['agreement'], generated['world_model'], generated['caption_model'])
#pprint.pprint(batch)


pprint.pprint(generated['world_model'])
print()
pprint.pprint(generated['caption_model'])
print()
pprint.pprint(generated['caption'])

str_captions = []

for i in range(generated['caption'].shape[0]):
    caption = ""
    for j in range(generated['caption'].shape[1]):
        caption += idx2word[generated['caption'][i][j]] + " "
    str_captions.append(caption)

pprint.pprint(str_captions)
pprint.pprint(generated['agreement'])
