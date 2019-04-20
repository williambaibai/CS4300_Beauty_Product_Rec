from . import *  
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Forever 4.300: Beauty Product Recommendation System"
net_id = "HaiYang Bai (hb388),\
		  Helen Liang (hl973),\
		  Joseph Kuo (jk2288),\
		  Yue Gao (yg98),\
		  Zidong Zheng (zz357)"

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

  def rating(self):
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
data = CustomUnpickler(open('data/data.p', 'rb')).load()
product_dict = data['product_dict']
category_dict = data['category_dict']
brand_dict = data['brand_dict']
brand_id_dict = data['brand_id_dict']
vectorizer = data['vectorizer']
id_to_idx = data['id_to_idx']
prod_vocab_mat = data['prod_vocab_mat']


@irsystem.route('/', methods=['GET'])
def search():
	# get search parameters
	brand = request.args.get('brand')
	category = request.args.get('category')
	price_range = request.args.get('price_range')
	skin_concern = request.args.getlist('skin_concern')
	skin_type = request.args.get('skin_type')
	
	if not skin_concern:
		return render_template('search.html', name=project_name, netid=net_id, output_message='', data=[])

	# Filter products by query category and brand 
	filtered_products_id = set(product_dict.keys())
	if category and str(category) != 'all_categories':
		filtered_products_id = filtered_products_id.intersection(set(category_dict[category]))
	if brand and str(brand) != 'all_brands':
		brand_name = brand_id_dict[brand]
		filtered_products_id = filtered_products_id.intersection(set(brand_dict[brand_name]))
	filtered_products_id = list(filtered_products_id)

	if len(filtered_products_id) == 0:
		return render_template('search.html', name=project_name, netid=net_id, output_message='No results for the selected Category and Brand, Please Try Again', data=[])

	# Use skin_concerns as query into the cosine sim search
	skin_concern_str = ''
	for word in skin_concern:
		skin_concern_str = skin_concern_str + word + " "
	query_vec = vectorizer.transform([skin_concern_str]).toarray()[0]

	filtered_mat = prod_vocab_mat[[id_to_idx[prod_id] for prod_id in filtered_products_id]]
	
	# Run Cosine Sim
	result_ids = cosine_sim(filtered_products_id, filtered_mat, query_vec)

	# Generate return data
	data = [{
		'name': product_dict[prod_id].name,
		'brand': product_dict[prod_id].brand,
		'image': product_dict[prod_id].image,
		'price': product_dict[prod_id].price,
		'rating': str(round(product_dict[prod_id].rating(), 2)),
		'description': product_dict[prod_id].description, 
		'sim_score': score
	} for (prod_id, score) in result_ids]
	
	return render_template('search.html', name=project_name, netid=net_id, output_message='Your Personalized Recommendation', data=data)


"""
Returns a sorted list of product ID most similar to query
"""
def cosine_sim(filtered_products_id, tfidf_mat, query_vec):
	result = [] 
	for i in range (0, len(tfidf_mat)):
		score = np.dot(query_vec, tfidf_mat[i])
		result.append(score)
	if len(result) < 21:
		sorted_idx = list(np.argsort(result))[::-1][:len(result)-1]
	else:
		sorted_idx = list(np.argsort(result))[::-1][:20]
	product_ids = [(filtered_products_id[idx], result[idx]) for idx in sorted_idx]
	return product_ids
