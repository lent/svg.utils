(function() {
	var attr = SVG.Element.prototype.attr;
	SVG.Element.prototype.attr = function() {
		if(this._absolutePathArray) {
			delete this._absolutePathArray;
		}
		return attr.apply(this, arguments);
	}
}())


var deltaTransformPoint = function(matrix, point)  {
	return [
		point[0] * matrix.a + point[1] * matrix.c,
		point[0] * matrix.b + point[1] * matrix.d
	];
};

SVG.RBox = function(element, relative) {
	/* initialize zero box */
	this.x = this.y = this.width = this.height = null;

	if (element) {
		var relative	= relative?relative.node:element.node.ownerSVGElement,
			matrix		= element.node.getTransformToElement(relative),
			box			= element.node.getBBox(),
			frame		= [[box.x, box.y], [box.x+box.width, box.y], [box.x+box.width, box.y+box.height], [box.x, box.y+box.height]],
			right		= null,
			bottom		= null;

		for(var i = 0; i<4; i++) {
			var p = SVG.utils.matrixTransform({
				x: frame[i][0],
				y: frame[i][1]
			}, matrix);
			if(this.x === null || p.x < this.x) {
				this.x = p.x;
			}
			if(this.y === null || p.y < this.y) {
				this.y = p.y;
			}
			if(right === null || p.x > right) {
				right = p.x;
			}
			if(bottom === null || p.y > bottom) {
				bottom = p.y;
			}
		}
		this.width	= right - this.x;
		this.height	= bottom - this.y;
	}

	/* add the center */
	this.cx = this.x + this.width  / 2;
	this.cy = this.y + this.height / 2;
}

SVG.extend(SVG.Element, {
	rbox: function(relative) {
		return new SVG.RBox(this, relative)
	}
});

SVG.Marker = SVG.invent({
	create: 'marker',
	inherit: SVG.Container,
	construct: {
		marker: function() {
			return this.defs().gradient(type, block)
		}
	}
});
SVG.extend(SVG.Defs, {
	// define marker 
	marker: function() {
		return this.put(new SVG.Marker);
	}
});

SVG.extend(SVG.Element, {
	_parseMatrix: function(o) {
		if(o.matrix && o.matrix !== SVG.defaults.matrix) {
			if(typeof o.matrix === "string") {
				var m = {};
				for(var i = 0, a = o.matrix.split(/[,\s]+/); i<a.length; i++) {
					m[String.fromCharCode(97+i)] = a[i];
				}
				o.matrix = m;
			}
			// calculate delta transform point
			var px    = deltaTransformPoint(o.matrix, [0, 1]),
				py    = deltaTransformPoint(o.matrix, [1, 0]),
				// calculate skew
				skewX = ((180 / Math.PI) * Math.atan2(px[1], px[0]) - 90),
				skewY = ((180 / Math.PI) * Math.atan2(py[1], py[0])),
				scaleX= Math.sqrt(o.matrix.a * o.matrix.a + o.matrix.b * o.matrix.b),
				scaleY= Math.sqrt(o.matrix.c * o.matrix.c + o.matrix.d * o.matrix.d);

			return {
				matrix:		SVG.defaults.matrix,
				x:			o.matrix.e,
				y:			o.matrix.f,
				scaleX:		scaleX,
				scaleY:		scaleY,
				//skewX:		skewX,
				//skewY:		skewY,
				rotation:	skewX, // rotation is the same as skew x
				cx:			0,
				cy:			0
			};
		}
		return o;
	},

	addClass: function(newCls) {
		var cls = this.node.getAttribute("class");
		//addClass / removeClass for svg
		if(!cls || !~cls.indexOf(newCls)) {
			this.node.setAttribute("class", (cls?cls+" ":"")+newCls);
		}
		return this;
	},
	removeClass: function(oldCls) {
		var cls = this.node.getAttribute("class");
		if(cls && ~cls.indexOf(oldCls)) {
			this.node.setAttribute("class", cls?cls.replace(new RegExp(oldCls+"($\|\\s)"), ""):"");
		}
		return this;
	},
	getPathArray: function() {
		var bbox	= this.node.getBBox(),
			rx		= this.attr('rx'),
			ry		= this.attr('ry'),
			w		= bbox.width,
			h		= bbox.height,
			x		= this.x(),
			y		= this.y();
		if(this instanceof SVG.G) {
			x = bbox.x;
			y = bbox.y;
		}
		if (rx || ry) {
			return [["M", x + rx, y], ["l", w - rx * 2, 0], ["a", rx, ry, 0, 0, 1, rx, ry], ["l", 0, h - ry * 2], ["a", rx, ry, 0, 0, 1, -rx, ry], ["l", rx * 2 - w, 0], ["a", rx, ry, 0, 0, 1, -rx, -ry], ["l", 0, ry * 2 - h], ["a", rx, ry, 0, 0, 1, rx, -ry], ["z"]];
		}
		return [["M", x, y], ["h", w], ["v", h], ["h", -w], ["z"]];
	},

	getAbsolutePathArray: function() {
		if(this._absolutePathArray) {
			return this._absolutePathArray;
		}
		return this._absolutePathArray = SVG.utils.toAbsolute(this.getPathArray());
	},
	
	getNearestPoint: function(x, y) {

		var matrix  = this.node.getTransformToElement(this.node.ownerSVGElement),
			arr     = this.getAbsolutePathArray(),
			c       = SVG.utils.toCurve(SVG.utils.transformAbsolutePathArray(arr, matrix), true);
		return SVG.utils.getNearestPoint(c, x, y);
	},

	getIntersections: function(c2, callback) {
		var matrix  = this.node.getTransformToElement(this.node.ownerSVGElement),
			arr     = this.getAbsolutePathArray(),
			c1      = SVG.utils.toCurve(SVG.utils.transformAbsolutePathArray(arr, matrix), true);

		if(c2 instanceof SVG.Element) {
			var bbox1   = this.rbox(),
				bbox2   = c2.rbox(),
				matrix  = c2.node.getTransformToElement(c2.node.ownerSVGElement),
				arr     = c2.getAbsolutePathArray();
			c2 = SVG.utils.toCurve(SVG.utils.transformAbsolutePathArray(arr, matrix), true);

			if(!c1.length || !c2.length) {
				return [];
			}

			// First check the bounds of the two paths. If they don't intersect,
			// we don't need to iterate through their curves.
			if(!(bbox1.x + bbox1.width >= bbox2.x && bbox1.y + bbox1.height >= bbox2.y && bbox1.x <= bbox2.x + bbox2.width && bbox1.y <= bbox2.y + bbox2.height)) {
				return callback ? callback([]) : [];
			}
		}

		return callback ?
			callback(SVG.utils.getIntersections(c1, c2)) :
			SVG.utils.getIntersections(c1, c2);
	}
});

SVG.extend(SVG.Ellipse, {
	getPathArray: function() {
		var x	= this.cx(),
			y	= this.cy(),
			rx	= this.attr('rx'),
			ry	= this.attr('ry');
		return [["M", x, y-ry], ["a", rx, ry, 0, 1, 1, 0, 2 * ry], ["a", rx, ry, 0, 1, 1, 0, -2 * ry], ["z"]];
	}
});

SVG.extend(SVG.Line, {
	getPathArray: function() {
		var x1	= this.attr("x1"),
			y1	= this.attr("y1"),
			x2	= this.attr('x2'),
			y2	= this.attr('y2');
		return [["M", x1, y1], ["L", x2, y2]];
	}
});

SVG.extend(SVG.Path, {
	getPathArray: function() {
		if(this.array) {
			return this.array.value;
		}
		var data = [];
		var length = this.node.pathSegList.numberOfItems;
		for(var i = 0; i < length; i++) {
			var pathSeg = this.node.pathSegList.getItem(i),
				segment = [pathSeg.pathSegTypeAsLetter],
				items	= ['r1', 'r2', 'angle', 'largeArcFlag', 'sweepFlag', 'x1', 'y1', 'x2', 'y2', 'x', 'y'],
				n;
			for(var k = 0; k<items.length; k++) {
				n = items[k];
				if(typeof pathSeg[n] !== 'undefined') {
					if(pathSeg[n] === true) {
						segment.push(1);
					} else if(pathSeg[n] === false) {
						segment.push(0);
					} else {
						segment.push(pathSeg[n]);
					}
				}
			}
			data.push(segment);
		}
		return data;
	}
});
