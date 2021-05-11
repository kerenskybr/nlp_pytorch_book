# Split the subset by rating to create new train, val, and test splits
by_rating = collections.defaultdict(list)
for _, row in review_subset.iterrows():
    by_rating[row.rating].append(row.to_dict())
# Create split data
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    n_test = int(args.test_proportion * n_total)
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'
    # Add to final list
    final_list.extend(item_list)
    
final_reviews = pd.DataFrame(final_list)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a­zA­Z.,!?]+", r" ", text)
    return text
final_reviews.review = final_reviews.review.apply(preprocess_text)