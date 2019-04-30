import torchbearer as tb

# Model keys
MU = tb.state_key('mu')
LOGVAR = tb.state_key('logvar')
MUV = tb.state_key('muv')
LOGVARV = tb.state_key('logvarv')
MUC = tb.state_key('muc')
LOGVARC = tb.state_key('logvarc')
V = tb.state_key('velocity')
Z = tb.state_key('shape')
C = tb.state_key('colour_code')
COMB = tb.state_key('combined_space')

# Image keys
MOVED = tb.state_key('moved')
VIS = tb.state_key('visu')
RECON = tb.state_key('recon')
RECON_TRAIN = tb.state_key('recon_train')
VISC = tb.state_key('visu_colour')
F2 = tb.state_key('f2')

# Dataloader keys
TL2 = tb.state_key('tl2')
VL2 = tb.state_key('vl2')
TL2I = tb.state_key('tl2i')
VL2I = tb.state_key('vl2i')

# Misc
CLASS = tb.state_key('class_labels')