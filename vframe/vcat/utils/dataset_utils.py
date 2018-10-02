
def create_splits(anno_index,randomize=True,split_val=0.2,rseed=1):
  """Convert annotation index into splits based on classes"""
  
  # TODO add num_split_vals
  annos_train = {}
  annos_valid = {}
  for class_idx, anno_obj in anno_index.items():
    n_annos = len(anno_obj['regions'])
    if randomize:
      random.seed(rseed)
      #random.shuffle(anno_obj['regions'])
    n_test = int(n_annos*split_val)
    class_annos_valid = anno_obj.copy()
    class_annos_valid['regions'] = anno_obj['regions'][:n_test]
    class_annos_train = anno_obj.copy()
    class_annos_train['regions'] = anno_obj['regions'][n_test:]
    annos_train[class_idx] = class_annos_train
    annos_valid[class_idx] = class_annos_valid

  return {'train':annos_train,'valid':annos_valid}