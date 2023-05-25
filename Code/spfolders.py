import splitfolders
#Split our folder in train val sets
splitfolders.ratio("Dataset3/", output="..output3", seed=1337, ratio=(.9, .1), group_prefix=None, move=False) # default values