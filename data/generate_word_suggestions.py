import json
import pickle

"""
Class Represetning the information of a product
"""
class Product:
  def __init__(self, name, brand, brand_id, image, description, price, category):
    self.name = name
    self.brand = brand
    self.brand_id = brand_id
    self.image = image
    self.description = description
    self.price = price
    self.category = category
    self.reviews = []
    self.rating = 0

  def compute_rating(self):
    total_score = 0
    for review in self.reviews:
      total_score += review.rating
    return total_score / len(self.reviews)

"""
Class Represetning the information of a review
"""
class Review:
  def __init__(self, text, rating, skin_type, skin_concerns):
    self.text = text
    self.rating = rating
    self.skin_type = skin_type
    self.skin_concerns = skin_concerns

class CustomUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if name == 'Product':
			return Product
		if name == 'Review':
			return Review
		return super().find_class(module, name)

# load data
data = CustomUnpickler(open('data.p', 'rb')).load()
product_dict = data['product_dict']
category_dict = data['category_dict']
brand_dict = data['brand_dict']
brand_id_dict = data['brand_id_dict']
skin_type_dict = data['skin_type_dict']
vectorizer = data['vectorizer']
id_to_idx = data['id_to_idx']
prod_vocab_mat = data['prod_vocab_mat']
words_compressed = data['words_compressed']
docs_compressed = data['docs_compressed']

word_suggestions = [{'name': word} for word in vectorizer.vocabulary_.keys()] 

with open('word_suggestions.json', 'w') as outfile:
  json.dump(word_suggestions, outfile)