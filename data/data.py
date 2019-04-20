import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# open csv files
cleanser_db = pd.read_csv('cleanser_db.csv')
eye_care_db = pd.read_csv('eye_care_db.csv')
lip_treatment_db = pd.read_csv('lip_treatment_db.csv')
masks_db = pd.read_csv('masks_db.csv')
moisturizer_db = pd.read_csv('moisturizer_db.csv')
sun_care_db = pd.read_csv('sun_care_db.csv')
treatment_db = pd.read_csv('treatment_db.csv')
"""
Class Representing the information of a product
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


def parse_category(category_name, db, product_dict, category_dict, brand_dict, brand_id_dict):
  for i in range(len(db.index)):
    product_id = str(db['product_id'][i])
    brand = str(db['brand'][i])
    brand_id = str(db['brand_id'][i])
    # handle special case for price
    price_str = str(db['price'][i])
    if price_str.find('-') == -1:
      price = float(price_str[1:])
    else:
      price = float(price_str[1:price_str.find('-')])
    if product_id not in product_dict:
      product_dict[product_id] = Product(str(db['name'][i]),
                                         brand,
                                         brand_id,
                                         str(db['product_image_url'][i]),
                                         str(db['description'][i]),
                                         price,
                                         category_name
                                        )

      if category_name not in category_dict:
        category_dict[category_name] = [product_id]
      else:
        category_dict[category_name].append(product_id)

      if brand not in brand_dict:
        brand_dict[brand] = [product_id]
      else:
        brand_dict[brand].append(product_id)

    if brand_id not in brand_id_dict:
      brand_id_dict[brand_id] = brand

    review = Review(str(db['review_text'][i]),
                    int(str(db['rating'][i])),
                    str(db['skin_type'][i]),
                    str(db['skin_concerns'][i])
                    )
    product_dict[product_id].reviews.append(review)

  return product_dict, category_dict, brand_dict, brand_id_dict

def parse_all():
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('cleanser', cleanser_db, {}, {}, {}, {})
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('eye_care', eye_care_db, product_dict, category_dict, brand_dict, brand_id_dict)
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('lip_treatment', lip_treatment_db, product_dict, category_dict, brand_dict, brand_id_dict)
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('mask', masks_db, product_dict, category_dict, brand_dict, brand_id_dict)
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('moisturizer', moisturizer_db, product_dict, category_dict, brand_dict, brand_id_dict)
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('sun_care', sun_care_db, product_dict, category_dict, brand_dict, brand_id_dict)
  product_dict, category_dict, brand_dict, brand_id_dict = parse_category('treatment', treatment_db, product_dict, category_dict, brand_dict, brand_id_dict)
  return product_dict, category_dict, brand_dict, brand_id_dict

product_dict, category_dict, brand_dict, brand_id_dict = parse_all()

def vectorize():
  prod_reviews = []
  product_info = list(product_dict.values())
  id_to_idx = {product_id : idx for (idx, product_id) in enumerate(product_dict.keys())}

  for product in product_info:
    concat_review = product.description
    for review in product.reviews:
      concat_review = concat_review + review.text
    prod_reviews.append(concat_review)

  vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.8, min_df=10, norm='l2')

  prod_vocab_mat = vectorizer.fit_transform(prod_reviews).toarray()

  return vectorizer, id_to_idx, prod_vocab_mat
  
vectorizer, id_to_idx, prod_vocab_mat = vectorize()

data = {'product_dict': product_dict, 
        'category_dict': category_dict, 
        'brand_dict': brand_dict, 
        'brand_id_dict': brand_id_dict, 
        'vectorizer': vectorizer, 
        'id_to_idx':id_to_idx, 
        'prod_vocab_mat': prod_vocab_mat}

pickle_file = 'data.p'

with open(pickle_file, "wb") as f:
  pickle.dump(data, f)