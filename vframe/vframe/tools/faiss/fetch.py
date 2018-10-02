
def fetch_features(net, weights):
    os.makedirs('nets', exist_ok=True)
    txt_fn = 'nets/' + net.lower() + '.txt'
    pkl_fn = 'nets/' + net.lower() + '.pkl'
    current_ts = ""
    last_modified = ""
    if os.path.exists(txt_fn):
        fh = open(txt_fn, "r") 
        current_ts = fh.readline()
        fh.close()
    print("Checking if we need to update...")
    url = s3_base_href + '/v1/metadata/features/' + net.lower() + '/index.pkl'
    print(url)
    try:
        request = urllib.request.Request(url, method='HEAD')
        response = urllib.request.urlopen(request)
        lines = str(response.info()).split('\n')
        for line in lines:
            partz = line.split(': ')
            if partz[0] == 'Last-Modified':
                last_modified = partz[1]
    except:
        print("/!\\ Error fetching pickle file!") 
        print(url)
    if current_ts == last_modified:
        print("Loading saved {} weights...".format(net))
        fh = open(pkl_fn, 'rb')
        raw = fh.read()
        fh.close()
        data = pickle.loads(raw)
    else:
        print("Fetching latest {} weights...".format(net))
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        raw = response.read()
        fh = open(txt_fn, 'w')
        fh.write(last_modified)
        fh.close()
        fh = open(pkl_fn, 'wb')
        fh.write(raw)
        fh.close()
        data = pickle.loads(raw)
    return data, fe

# data, fe = fetch_features(net='VGG16', weights='imagenet')

# fe = FeatureExtractor(net='VGG16', weights='imagenet')
