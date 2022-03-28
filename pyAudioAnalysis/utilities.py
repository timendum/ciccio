import numpy


def peakdet(v, delta):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    v = numpy.asarray(v)

    if not numpy.isscalar(delta):
        raise ValueError("Input argument delta must be a scalar")

    if delta <= 0:
        raise ValueError("Input argument delta must be positive")

    mn, mx = numpy.Inf, -numpy.Inf
    mnpos, mxpos = numpy.NaN, numpy.NaN

    lookformax = True

    for i, vi in enumerate(v):
        if vi > mx:
            mx = vi
            mxpos = i
        if vi < mn:
            mn = vi
            mnpos = i

        if lookformax:
            if vi < mx - delta:
                maxtab.append(mxpos)
                mn = vi
                mnpos = i
                lookformax = False
        else:
            if vi > mn + delta:
                mintab.append(mnpos)
                mx = vi
                mxpos = i
                lookformax = True

    return numpy.array(maxtab), numpy.array(mintab)
