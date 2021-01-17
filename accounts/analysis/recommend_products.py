from celery import shared_task
from .utils import send_email
import numpy as np
import pandas as pd
import math
import lightgbm as lgb


IDIR = 'inputs/'
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])

apriori = pd.read_csv(IDIR+'Apriori.csv')
user_top5_aisle_product = pd.read_csv(IDIR+'user_top5_aisle_product.csv')

ordersTrain=orders[orders['eval_set']=='train']
train_usr = pd.DataFrame(ordersTrain['user_id'].unique(), columns =['user_id'])
train_usr_sample =  train_usr.sample(2500, random_state=42).sort_values('user_id').reset_index(drop = True)

prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
del prods

orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

'''
Creating user level features
'''

usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
users['user_max_order_num'] =  priors.groupby('user_id')['order_number'].max().astype(np.int16)
users['total_buy_max'] =  priors.groupby(['user_id','product_id'])['product_id'].count().reset_index(level = 'user_id').reset_index(drop = True).groupby('user_id').max().astype(np.int16)
users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)

priors['user_product'] = priors.product_id + priors.user_id * 100000

d= dict()
for row in priors.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                min(d[z][2], (row.order_number, row.order_id)),
                d[z][3] + row.add_to_cart_order)

userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d

userXproduct.columns = ['nb_orders', 'last_order_id','first_order_number', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.first_order_number = userXproduct.first_order_number.map(lambda x: x[0]).astype(np.int16)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
del priors


train_orders = orders[orders.eval_set == 'train']
train_orders_train = train_orders[~train_orders.user_id.isin(train_usr_sample['user_id'])]
train_orders_cv = train_orders[train_orders.user_id.isin(train_usr_sample['user_id'])]
train_orders_product  = train_orders_cv.merge(train, how = 'inner', left_on = 'order_id', right_on = 'order_id')

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list

    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)
    df['user_total_buy_max'] = df.user_id.map(users.total_buy_max).astype(np.int16)

    print('order related features')
    df['order_dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = (df.days_since_prior_order / df.user_average_days_between_orders).map(
        lambda x: 0 if math.isnan(x) else x).astype(np.float32)

    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate).astype(np.float32)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    # df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = ((df.UP_orders - 1) / (df.user_total_orders - 1).astype(np.float32))
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(
        lambda x: min(x, 24 - x)).astype(np.int8)
    df['UP_delta_dow_vs_last'] = abs(df.order_dow - df.UP_last_order_id.map(orders.order_dow)).map(
        lambda x: min(x, 7 - x)).astype(np.int8)
    df['UP_drop_chance'] = (df.user_total_orders - df.UP_last_order_id.map(orders.order_number)).astype(np.float)
    df['UP_chance_vs_bought'] = (df.user_total_orders - df.z.map(userXproduct.first_order_number)).astype(np.float32)
    df['UP_chance'] = (df.UP_orders - 1) / (df.user_total_orders - df.z.map(userXproduct.first_order_number)).astype(
        np.float32)
    df['UP_chance_ratio'] = (
                1 / (df.user_total_orders - df.UP_last_order_id.map(orders.order_number)) - (df.UP_orders - 1) / (
                    df.user_total_orders - df.z.map(userXproduct.first_order_number))).astype(np.float32)
    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    # df.drop(['order_id','product_id'], axis=1)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)

train_orders_train = train_orders[~train_orders.user_id.isin(train_usr_sample['user_id'])]
train_orders_cv = train_orders[train_orders.user_id.isin(train_usr_sample['user_id'])]

df_train, labels_train = features(train_orders_train, labels_given=True)
df_cv, labels_cv = features(train_orders_cv, labels_given=True)

features_to_use = ['user_total_orders', 'user_total_items',
       'total_distinct_items', 'user_average_days_between_orders',
       'user_average_basket', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last', 'UP_delta_dow_vs_last', 'UP_drop_chance',
       'UP_chance_vs_bought', 'user_total_buy_max', 'UP_chance', 'UP_chance_ratio','aisle_id']

'''formating for lgb'''

d_train = lgb.Dataset(df_train[features_to_use],
                      label=labels_train,
                      categorical_feature=['aisle_id', 'department_id'])

#Best Parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

#train
bst = lgb.train(params, d_train, ROUNDS)


def pred_fo(df_order, order_id):
    preds = bst.predict(df_order[features_to_use])
    df_order['pred'] = preds
    d = dict()
    for row in df_order.itertuples():
        if row.pred > 0.22:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in df_order.order_id:
        if order not in d:
            d[order] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')

    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']

    actual = list([i.split(' ') for i in list(sub['products'])])
    predicted = list(train_orders_product[train_orders_product['order_id'] == order_id]['product_id'])
    acc = 0
    rr = [val for val in predicted if val in actual]
    if len(rr) == 0:
        acc = 0
    else:
        acc = 1

    return (acc)

@shared_task
def recommend(orderid):
    df_cv_l = df_cv
    df_cv_l['label'] = labels_cv
    df_order = df_cv_l[df_cv_l['order_id'] == orderid]
    accuracy = pred_fo(df_order, orderid)

    df_ol = df_cv_l[df_cv_l['order_id'] == orderid]
    df_op = df_ol[df_ol['label'] == 1]
    products = list(train_orders_product[train_orders_product['order_id'] == orderid]['product_id'])
    user = list(df_op['user_id'].unique())
    user_products_cluster = user_top5_aisle_product[user_top5_aisle_product.user_id.isin(df_op['user_id'].unique())]
    upc_product = list(user_products_cluster['product1'])
    upc_product.append(list(user_products_cluster['product2']))
    for i in products:
        temp = apriori[apriori['item_A'] == i].nlargest(9, 'lift')
        p_products = list(temp['item_B'])
        p_products.append(upc_product)
        rr = [val for val in p_products if val in products]
        if len(rr) > 0:
            accuracy = accuracy + 1

    print(accuracy / len(products))
    #products = {"product1": products[0], "product2": products[1], "product3": products[2]}
    products = {"product1": "banana", "product2": "banana", "product3": "banana"}
    send_email(data=products)
