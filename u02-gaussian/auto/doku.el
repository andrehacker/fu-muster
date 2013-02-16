(TeX-add-style-hook "doku"
 (lambda ()
    (LaTeX-add-labels
     "picture-label")
    (TeX-add-symbols
     "im"
     "rg"
     "ggt")
    (TeX-run-style-hooks
     "polynom"
     "verbatim"
     "amsmath"
     "amssymb"
     "hyperref"
     "graphicx"
     "inputenc"
     "utf8"
     "babel"
     "ngerman"
     "latex2e"
     "art10"
     "article")))

