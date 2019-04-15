from . import *  
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import product_dict, category_dict, brand_dict, brand_id_dict
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Forever 4.300: Beauty Product Recommendation System"
net_id = "HaiYang Bai (hb388),\
		  Helen Liang (hl973),\
		  Joseph Kuo (jk2288)],\
		  Yue Gao (yg98),\
		  Zidong Zheng (zz357)"

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
				
	# Vectorize product data along with product ID for cosine sim
	filtered_products_id = list(filtered_products_id)
	filtered_products = [product_dict[prod_id] for prod_id in filtered_products_id]
	filtered_prod_reviews = []
	if str(skin_type) == 'all_skin_types':
		for product in filtered_products:
			concat_review = ''
			for review in product.reviews:
				concat_review = concat_review + review.text
			filtered_prod_reviews.append(concat_review)
	else:
		# only consider reviews of query skin type or no skin type
		for product in filtered_products:
			concat_review = ''
			for review in product.reviews:
				if review.skin_type == skin_type or review.skin_type == '':
					concat_review = concat_review + review.text
			filtered_prod_reviews.append(concat_review)
	
	# Use skin_concerns as query into the cosine sim search
	skin_concern_str = ''
	for word in skin_concern:
		skin_concern_str = skin_concern_str + word + " "
	filtered_products_id.insert(0, 'query')
	filtered_prod_reviews.insert(0, skin_concern_str)

	vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.8, min_df=0.1, norm='l2')
	try:
		prod_by_vocab = vectorizer.fit_transform(filtered_prod_reviews).toarray()
	except:
		print(len(filtered_prod_reviews)) 
		return render_template('search.html', name=project_name, netid=net_id, output_message='No results for the selected Category and Brand, Please Try Again', data=[])

	# Run Cosine Sim
	result_ids = cosine_sim(filtered_products_id, prod_by_vocab)

	# Generate return data
	data = [{
		'name': product_dict[prod_id].name,
		'brand': product_dict[prod_id].brand,
		'image': product_dict[prod_id].image,
		'price': product_dict[prod_id].price,
		'rating': product_dict[prod_id].rating()
	} for prod_id in result_ids]
	
	return render_template('search.html', name=project_name, netid=net_id, output_message='Your Personalized Recommendation', data=data)


"""
Returns a sorted list of product ID most similar to query
"""
# TODO: Save sim scores
def cosine_sim(filtered_products_id, tfidf_mat):
	result = [0] # similarity with itself is 0
	for i in range (1, len(tfidf_mat)):
		score = np.dot(tfidf_mat[0], tfidf_mat[i])
		result.append(score)
	if len(result) < 21:
		sorted_idx = list(np.argsort(result))[::-1][:len(result)-1]
	else:
		sorted_idx = list(np.argsort(result))[::-1][:20]
	product_ids = [filtered_products_id[idx] for idx in sorted_idx]
	if 'query' in product_ids:
		product_ids.remove('query')
	return product_ids
