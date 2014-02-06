svg.utils
=========

some functions to extend svg.js and general Curve algorithms


### svg.ext.js
svg.ext.js extends SVG.RBox to being able to relate the box to a specific element (fallback owner SVG)
* Adds SVG.Marker
* an experimental parse Matrix
* css class function addClass / removeClass. (jQuery doesn't work on SVG)
* getPathArray function gets an array representation of the path segments in the form [['command', arguments...], ['command', arguments...], ...]
* getNearestPoint calculates the Neareast point to the x/y given to the border of the element
* getIntersections calculates intersections to a of the element with another given element or array of curves.

### svg.utils.js
functions for path arrays and curves, similar to raphael.js and paper.js internals
