from dsl import *

####startfunctions####

def verify_007bbfb7(I: Grid) -> Grid:
        return compose((lbind(paint, I)), (chain((chain(first, (rbind(rapply, (dmirror(I)))), initset)), (rbind(compose, asobject)), (chain((lbind(rbind, sfilter)), (lbind(compose, flip)), (lbind(matcher, first)))))))


####endfunctions####