from shapeworld import dataset
import pprint

dataset = dataset(dtype='agreement', name='oneshape_simple_textselect', )
generated = dataset.generate(n=30, mode='train', noise_range=0.1, include_model=True)
