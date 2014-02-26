/* svg.utils.js
 * Copyright (c) 2013, Axel Meinhardt 
 *
 * Distributed under the MIT license. See LICENSE file for details.
 *
 * All rights reserved.
 *
 ***
 * Thanks to:
 *
 * Paper.js v0.9.15 - The Swiss Army Knife of Vector Graphics Scripting.
 * http://paperjs.org/
 *
 * Copyright (c) 2011 - 2013, Juerg Lehni & Jonathan Puckey
 * http://lehni.org/ & http://jonathanpuckey.com/
 *
 * Distributed under the MIT license. See LICENSE file for details.
 *
 * All rights reserved.
 *
 * Date: Sun Dec 1 23:54:52 2013 +0100
 *
 */

var
epsilon = 10e-12,
//abscissae and weights for performing Legendre-Gauss quadrature integral approximation
abscissas = [
	[  0.5773502691896257645091488],
	[0,0.7745966692414833770358531],
	[  0.3399810435848562648026658,0.8611363115940525752239465],
	[0,0.5384693101056830910363144,0.9061798459386639927976269],
	[  0.2386191860831969086305017,0.6612093864662645136613996,0.9324695142031520278123016],
	[0,0.4058451513773971669066064,0.7415311855993944398638648,0.9491079123427585245261897],
	[  0.1834346424956498049394761,0.5255324099163289858177390,0.7966664774136267395915539,0.9602898564975362316835609],
	[0,0.3242534234038089290385380,0.6133714327005903973087020,0.8360311073266357942994298,0.9681602395076260898355762],
	[  0.1488743389816312108848260,0.4333953941292471907992659,0.6794095682990244062343274,0.8650633666889845107320967,0.9739065285171717200779640],
	[0,0.2695431559523449723315320,0.5190961292068118159257257,0.7301520055740493240934163,0.8870625997680952990751578,0.9782286581460569928039380],
	[  0.1252334085114689154724414,0.3678314989981801937526915,0.5873179542866174472967024,0.7699026741943046870368938,0.9041172563704748566784659,0.9815606342467192506905491],
	[0,0.2304583159551347940655281,0.4484927510364468528779129,0.6423493394403402206439846,0.8015780907333099127942065,0.9175983992229779652065478,0.9841830547185881494728294],
	[  0.1080549487073436620662447,0.3191123689278897604356718,0.5152486363581540919652907,0.6872929048116854701480198,0.8272013150697649931897947,0.9284348836635735173363911,0.9862838086968123388415973],
	[0,0.2011940939974345223006283,0.3941513470775633698972074,0.5709721726085388475372267,0.7244177313601700474161861,0.8482065834104272162006483,0.9372733924007059043077589,0.9879925180204854284895657],
	[  0.0950125098376374401853193,0.2816035507792589132304605,0.4580167776572273863424194,0.6178762444026437484466718,0.7554044083550030338951012,0.8656312023878317438804679,0.9445750230732325760779884,0.9894009349916499325961542]
],
weights = [
	[1],
	[0.8888888888888888888888889,0.5555555555555555555555556],
	[0.6521451548625461426269361,0.3478548451374538573730639],
	[0.5688888888888888888888889,0.4786286704993664680412915,0.2369268850561890875142640],
	[0.4679139345726910473898703,0.3607615730481386075698335,0.1713244923791703450402961],
	[0.4179591836734693877551020,0.3818300505051189449503698,0.2797053914892766679014678,0.1294849661688696932706114],
	[0.3626837833783619829651504,0.3137066458778872873379622,0.2223810344533744705443560,0.1012285362903762591525314],
	[0.3302393550012597631645251,0.3123470770400028400686304,0.2606106964029354623187429,0.1806481606948574040584720,0.0812743883615744119718922],
	[0.2955242247147528701738930,0.2692667193099963550912269,0.2190863625159820439955349,0.1494513491505805931457763,0.0666713443086881375935688],
	[0.2729250867779006307144835,0.2628045445102466621806889,0.2331937645919904799185237,0.1862902109277342514260976,0.1255803694649046246346943,0.0556685671161736664827537],
	[0.2491470458134027850005624,0.2334925365383548087608499,0.2031674267230659217490645,0.1600783285433462263346525,0.1069393259953184309602547,0.0471753363865118271946160],
	[0.2325515532308739101945895,0.2262831802628972384120902,0.2078160475368885023125232,0.1781459807619457382800467,0.1388735102197872384636018,0.0921214998377284479144218,0.0404840047653158795200216],
	[0.2152638534631577901958764,0.2051984637212956039659241,0.1855383974779378137417166,0.1572031671581935345696019,0.1215185706879031846894148,0.0801580871597602098056333,0.0351194603317518630318329],
	[0.2025782419255612728806202,0.1984314853271115764561183,0.1861610000155622110268006,0.1662692058169939335532009,0.1395706779261543144478048,0.1071592204671719350118695,0.0703660474881081247092674,0.0307532419961172683546284],
	[0.1894506104550684962853967,0.1826034150449235888667637,0.1691565193950025381893121,0.1495959888165767320815017,0.1246289712555338720524763,0.0951585116824927848099251,0.0622535239386478928628438,0.0271524594117540948517806]
],

touches = function(rect1, rect2) {
	return rect1.x + rect1.width >= rect2.x && rect1.y + rect1.height >= rect2.y && rect1.x <= rect2.x + rect2.width && rect1.y <= rect2.y + rect2.height;
},
distance2 = function(p, x, y)  {
	var x = p[0] - x,
		y = p[1] - y;
	return x*x + y*y;
},
integrate =  function(f, a, b, n) {
	var x = abscissas[n - 2],
		w = weights[n - 2],
		A = 0.5 * (b - a),
		B = A + a,
		i = 0,
		m = (n + 1) >> 1,
		sum = n & 1 ? w[i++] * f(B) : 0; 
	while (i < m) {
		var Ax = A * x[i];
		sum += w[i++] * (f(B + Ax) + f(B - Ax));
	}
	return A * sum;
},
// http://schepers.cc/getting-to-the-point
catmullRom2bezier = function(crp, z) {
	var d = [];
	for(var i = 0, iLen = crp.length; iLen - 2 * !z > i; i += 2) {
		var p = [{
			x: +crp[i - 2],
			y: +crp[i - 1]
		}, {
			x: +crp[i],
			y: +crp[i + 1]
		}, {
			x: +crp[i + 2],
			y: +crp[i + 3]
		}, {
			x: +crp[i + 4],
			y: +crp[i + 5]
		}];
		if(z) {
			if(!i) {
				p[0] = {
					x: +crp[iLen - 2],
					y: +crp[iLen - 1]
				};
			} else if(iLen - 4 == i) {
				p[3] = {
					x: +crp[0],
					y: +crp[1]
				};
			} else if(iLen - 2 == i) {
				p[2] = {
					x: +crp[0],
					y: +crp[1]
				};
				p[3] = {
					x: +crp[2],
					y: +crp[3]
				};
			}
		} else {
			if(iLen - 4 == i) {
				p[3] = p[2];
			} else if(!i) {
				p[0] = {
					x: +crp[i],
					y: +crp[i + 1]
				};
			}
		}
		d.push(["C", (-p[0].x + 6 * p[1].x + p[2].x) / 6, (-p[0].y + 6 * p[1].y + p[2].y) / 6, (p[1].x + 6 * p[2].x - p[3].x) / 6, (p[1].y + 6 * p[2].y - p[3].y) / 6,
			p[2].x,
			p[2].y
		]);
	}
	return d;
},
rotate = function(x, y, rad) {
	var X = x * Math.cos(rad) - y * Math.sin(rad),
		Y = x * Math.sin(rad) + y * Math.cos(rad);
	return {
		x: X,
		y: Y
	};
},
matrixTransform = function(point, matrix) {
	var x = point.x,
		y = point.y;
	point.x = matrix.a * x + matrix.c * y + matrix.e;
	point.y = matrix.b * x + matrix.d * y + matrix.f;
	return point;
},
// function arcTransform() is needed to flatten transformations of elliptical arcs
// Note! This is not used if paths are normalized
arcTransform = function(a_rh, a_rv, a_offsetrot, largeArcFlag, sweepFlag, endpoint, matrix) {
	function NEARZERO(B) {
		if(Math.abs(B) < 1e-16) {
			return true;
		}
		else {
			return false;
		}
	}

	var m = [], // matrix representation of transformed ellipse
		A, B, C, // ellipse implicit equation:
		ac, A2, C2, // helpers for angle and halfaxis-extraction.
		rh  = a_rh,
		rv  = a_rv,
		rot = a_offsetrot = a_offsetrot * (Math.PI / 180), // deg->rad
		s   = parseFloat(Math.sin(rot)), // sin and cos helpers (the former offset rotation)
		c   = parseFloat(Math.cos(rot));

	// build ellipse representation matrix (unit circle transformation).
	// the 2x2 matrix multiplication with the upper 2x2 of a_mat is inlined.
	m[0] = matrix.a * +rh * c + matrix.c * rh * s;
	m[1] = matrix.b * +rh * c + matrix.d * rh * s;
	m[2] = matrix.a * -rv * s + matrix.c * rv * c;
	m[3] = matrix.b * -rv * s + matrix.d * rv * c;

	// to implict equation (centered)
	A = (m[0] * m[0]) + (m[2] * m[2]);
	C = (m[1] * m[1]) + (m[3] * m[3]);
	B = (m[0] * m[1] + m[2] * m[3]) * 2.0;

	// precalculate distance A to C
	ac = A - C;

	// convert implicit equation to angle and halfaxis:
	if(NEARZERO(B)) {
		a_offsetrot = 0;
		A2 = A;
		C2 = C;
	} else {
		if(NEARZERO(ac)) {
			A2 = A + B * 0.5;
			C2 = A - B * 0.5;
			a_offsetrot = Math.PI / 4.0;
		} else {
			// Precalculate radical:
			var K = 1 + B * B / (ac * ac);

			// Clamp (precision issues might need this.. not likely, but better save than sorry)
			if(K < 0) {
				K = 0;
			} else {
				K = Math.sqrt(K);
			}

			A2 = 0.5 * (A + C + K * ac);
			C2 = 0.5 * (A + C - K * ac);
			a_offsetrot = 0.5 * Math.atan2(B, ac);
		}
	}

	// This can get slightly below zero due to rounding issues.
	// it's save to clamp to zero in this case (this yields a zero length halfaxis)
	if(A2 < 0) {
		A2 = 0;
	} else {
		A2 = Math.sqrt(A2);
	}
	if(C2 < 0) {
		C2 = 0;
	} else {
		C2 = Math.sqrt(C2);
	}

	// now A2 and C2 are half-axis:
	if(ac <= 0) {
		a_rv = A2;
		a_rh = C2;
	} else {
		a_rv = C2;
		a_rh = A2;
	}

	// If the transformation matrix contain a mirror-component 
	// winding order of the ellise needs to be changed.
	if((matrix.a * matrix.d) - (matrix.b * matrix.c) < 0) {
		if(!sweepFlag) sweepFlag = 1;
		else sweepFlag = 0;
	}

	// Finally, transform arc endpoint. This takes care about the
	// translational part which we ignored at the whole math-showdown above.
	matrixTransform(endpoint, matrix);

	// Radians back to degrees
	a_offsetrot = a_offsetrot * 180 / Math.PI;

	var r = ["A", a_rh, a_rv, a_offsetrot, largeArcFlag, sweepFlag, endpoint.x, endpoint.y];
	return r;
},
//only used for solveQuadratic/Cubic
addRoot = function(root, roots, count, min, max) {
	if(min === undefined || root >= min-epsilon && root <= max+epsilon)
		roots[count++] = root<min?min: (root>max?max:root);
	return count;
},
solveQuadratic = function(a, b, c, roots, min, max) {
	var count = 0;

	if(Math.abs(a) < epsilon) {
		if(Math.abs(b) >= epsilon) {
			return addRoot(-c / b, roots, count, min, max);
		}
		return Math.abs(c) < epsilon ? -1 : 0;
	}
	var p = b / (2 * a);
	var q = c / a;
	var p2 = p * p;
	if(p2 < q - epsilon) {
		return 0;
	}
	var s = p2 > q ? Math.sqrt(p2 - q) : 0;
	count = addRoot(s - p, roots, count, min, max);
	if(s > 0) {
		addRoot(-s - p, roots, count, min, max);
	}
	return count;
},
solveCubic = function(v, coord, val, roots, min, max) {
	var p1 = v[coord],
		c1 = v[coord + 2],
		c2 = v[coord + 4],
		p2 = v[coord + 6],
		c = 3 * (c1 - p1),
		b = 3 * (c2 - c1) - c,
		a = p2 - p1 - c - b,
		d = p1 - val;
	if(Math.abs(a) < epsilon)
		return solveQuadratic(b, c, d, roots, min, max);

	var count = 0;

	b /= a;
	c /= a;
	d /= a;
	var bb = b * b,
		p = (bb - 3 * c) / 9,
		q = (2 * bb * b - 9 * b * c + 27 * d) / 54,
		ppp = p * p * p,
		D = q * q - ppp;
	b /= 3;
	if(Math.abs(D) < epsilon) {
		if(Math.abs(q) < epsilon)
			return addRoot(-b, roots, count, min, max);
		var sqp = Math.sqrt(p),
			snq = q > 0 ? 1 : -1;
		addRoot(-snq * 2 * sqp - b, roots, count, min, max);
		return addRoot(snq * sqp - b, roots, count, min, max);
	}
	if(D < 0) {
		var sqp = Math.sqrt(p),
			phi = Math.acos(q / (sqp * sqp * sqp)) / 3,
			t = -2 * sqp,
			o = 2 * Math.PI / 3;
		addRoot(t * Math.cos(phi) - b, roots, count, min, max);
		addRoot(t * Math.cos(phi + o) - b, roots, count, min, max);
		return addRoot(t * Math.cos(phi - o) - b, roots, count, min, max);
	}
	var A = (q > 0 ? -1 : 1) * Math.pow(Math.abs(q) + Math.sqrt(D), 1 / 3);
	return addRoot(A + p / A - b, roots, count, min, max);
},
getSignedLineDistance = function(px, py, vx, vy, x, y) {
	vx -= px;
	vy -= py;
	var m = vy / vx,
		b = py - m * px;
	return(y - (m * x) - b) / Math.sqrt(m * m + 1);
},
getConvexHull = function(dq0, dq1, dq2, dq3) {
	var p0 = [0, dq0],
		p1 = [1 / 3, dq1],
		p2 = [2 / 3, dq2],
		p3 = [1, dq3],
		dist1 = getSignedLineDistance(0, dq0, 1, dq3, 1 / 3, dq1),
		dist2 = getSignedLineDistance(0, dq0, 1, dq3, 2 / 3, dq2);
	if(dist1 * dist2 < 0) {
		return [p0, p1, p3, p2];
	}
	var pmax, cross;
	if(Math.abs(dist1) > Math.abs(dist2)) {
		pmax = p1;
		cross = (dq3 - dq2 - (dq3 - dq0) / 3) * (2 * (dq3 - dq2) - dq3 + dq1) / 3;
	} else {
		pmax = p2;
		cross = (dq1 - dq0 + (dq0 - dq3) / 3) * (-2 * (dq0 - dq1) + dq0 - dq2) / 3;
	}
	return cross < 0 ? [p0, pmax, p3] : [p0, p1, p2, p3];
},
clipFatLine = function(v1, v2, range2) {
	var p0x = v1[0],
		p0y = v1[1],
		p1x = v1[2],
		p1y = v1[3],
		p2x = v1[4],
		p2y = v1[5],
		p3x = v1[6],
		p3y = v1[7],
		q0x = v2[0],
		q0y = v2[1],
		q1x = v2[2],
		q1y = v2[3],
		q2x = v2[4],
		q2y = v2[5],
		q3x = v2[6],
		q3y = v2[7],
		d1 = getSignedLineDistance(p0x, p0y, p3x, p3y, p1x, p1y) || 0,
		d2 = getSignedLineDistance(p0x, p0y, p3x, p3y, p2x, p2y) || 0,
		factor = d1 * d2 > 0 ? 3 / 4 : 4 / 9,
		dmin = factor * Math.min(0, d1, d2),
		dmax = factor * Math.max(0, d1, d2),
		dq0 = getSignedLineDistance(p0x, p0y, p3x, p3y, q0x, q0y),
		dq1 = getSignedLineDistance(p0x, p0y, p3x, p3y, q1x, q1y),
		dq2 = getSignedLineDistance(p0x, p0y, p3x, p3y, q2x, q2y),
		dq3 = getSignedLineDistance(p0x, p0y, p3x, p3y, q3x, q3y);
	if(dmin > Math.max(dq0, dq1, dq2, dq3) || dmax < Math.min(dq0, dq1, dq2, dq3)) {
		return 0;
	}
	var hull = getConvexHull(dq0, dq1, dq2, dq3),
		swap;
	if(dq3 < dq0) {
		swap = dmin;
		dmin = dmax;
		dmax = swap;
	}
	var tmaxdmin = -Infinity,
		tmin = Infinity,
		tmax = -Infinity;
	for(var i = 0, l = hull.length; i < l; i++) {
		var p1 = hull[i],
			p2 = hull[(i + 1) % l];
		if(p2[1] < p1[1]) {
			swap = p2;
			p2 = p1;
			p1 = swap;
		}
		var x1 = p1[0],
			y1 = p1[1],
			x2 = p2[0],
			y2 = p2[1];
		var inv = (y2 - y1) / (x2 - x1);
		if(dmin >= y1 && dmin <= y2) {
			var ixdx = x1 + (dmin - y1) / inv;
			if(ixdx < tmin)
				tmin = ixdx;
			if(ixdx > tmaxdmin)
				tmaxdmin = ixdx;
		}
		if(dmax >= y1 && dmax <= y2) {
			var ixdx = x1 + (dmax - y1) / inv;
			if(ixdx > tmax)
				tmax = ixdx;
			if(ixdx < tmin)
				tmin = 0;
		}
	}
	if(tmin !== Infinity && tmax !== -Infinity) {
		var min = Math.min(dmin, dmax),
			max = Math.max(dmin, dmax);
		if(dq3 > min && dq3 < max)
			tmax = 1;
		if(dq0 > min && dq0 < max)
			tmin = 0;
		if(tmaxdmin > tmax)
			tmax = 1;
		var v2tmin = range2[0],
			tdiff = range2[1] - v2tmin;
		range2[0] = v2tmin + tmin * tdiff;
		range2[1] = v2tmin + tmax * tdiff;
		if((tdiff - (range2[1] - range2[0])) / tdiff >= 0.2)
			return 1;
	}

	var rect1 = Curve.getBounds(v1),
		rect2 = Curve.getBounds(v2);
	return(rect1.x + rect1.width >= rect2.x && rect1.y + rect1.height >= rect2.y && rect1.x <= rect2.x + rect2.width && rect1.y <= rect2.y + rect2.height) ? -1 : 0;
};

var Curve = {
	isLinear: function(v) {
		return v[0] === v[2] && v[1] === v[3] &&
			v[4] === v[6] && v[5] === v[7];
	},
	getBounds: function(v) {
		var min = v.slice(0, 2),
			max = min.slice(),
			roots = [0, 0],
			tMin = 0.00001,
			tMax = 1 - tMin,
			v0, v1, v2, v3, a, b, c, count;
		for(var coord = 0; coord < 2; coord++) {
			v0	= v[coord];
			v1	= v[coord+2];
			v2	= v[coord+4];
			v3	= v[coord+6];
			a	= 3 * (v1 - v2) - v0 + v3;
			b	= 2 * (v0 + v2) - 4 * v1;
			c	= v1 - v0;
			count = solveQuadratic(a, b, c, roots);
			min[coord] = Math.min(v3, min[coord]);
			max[coord] = Math.max(v3, max[coord]);
			for(var i = 0; i < count; i++) {
				var t = roots[i],
				u = 1 - t;
				if(tMin < t && t < tMax) {
					var w = Math.pow(u,3) * v0 + 3 * Math.pow(u,2) * t * v1 + 3 * u * Math.pow(t,2) * v2 + Math.pow(t,3) * v3;
					min[coord] = Math.min(w, min[coord]);
					max[coord] = Math.max(w, max[coord]);
				}
			}
		}
		return {
			x: min[0],
			y: min[1],
			width: max[0] - min[0],
			height: max[1] - min[1]
		};
	},
	evaluate: function(v, t) {
		var p1x = v[0],
			p1y = v[1],
			c1x = v[2],
			c1y = v[3],
			c2x = v[4],
			c2y = v[5],
			p2x = v[6],
			p2y = v[7],
			x, y;

		if(t === 0 || t === 1) {
			x = t === 0 ? p1x : p2x;
			y = t === 0 ? p1y : p2y;
		} else {
			var cx = 3 * (c1x - p1x),
				bx = 3 * (c2x - c1x) - cx,
				ax = p2x - p1x - cx - bx,

				cy = 3 * (c1y - p1y),
				by = 3 * (c2y - c1y) - cy,
				ay = p2y - p1y - cy - by;
				x = ((ax * t + bx) * t + cx) * t + p1x;
				y = ((ay * t + by) * t + cy) * t + p1y;
		}
		return [x, y];
	},
	getParameterOf: function(v, x, y) {
		if(Math.abs(v[0] - x) < 0.00001 && Math.abs(v[1] - y) < 0.00001)
			return 0;
		if(Math.abs(v[6] - x) < 0.00001 && Math.abs(v[7] - y) < 0.00001)
			return 1;
		var txs = [],
			tys = [],
			sx = solveCubic(v, 0, x, txs),
			sy = solveCubic(v, 1, y, tys),
			tx, ty;
		for(var cx = 0; sx == -1 || cx < sx;) {
			if(sx == -1 || (tx = txs[cx++]) >= 0 && tx <= 1) {
				for(var cy = 0; sy == -1 || cy < sy;) {
					if(sy == -1 || (ty = tys[cy++]) >= 0 && ty <= 1) {
						if(sx == -1) {
							tx = ty;
						} else if(sy == -1) {
							ty = tx;
						}
						if(Math.abs(tx - ty) < 0.00001) {
							return(tx + ty) * 0.5;
						}
					}
				}
				if(sx == -1) {
					break;
				}
			}
		}
		return null;
	},

	getLineIntersection: function(v1, v2) {
		var l1sx = v1[0],
			l1sy = v1[1],
			l1ex = v1[6],
			l1ey = v1[7],
			l2sx = v2[0],
			l2sy = v2[1],
			l2ex = v2[6],
			l2ey = v2[7];
		//shift back to 0
		l1ex -= l1sx;
		l1ey -= l1sy;
		l2ex -= l2sx;
		l2ey -= l2sy;
		var cross = l2ey * l1ex - l2ex * l1ey;
		if(Math.abs(cross) > 10e-12) {
			var dx = l1sx - l2sx,
				dy = l1sy - l2sy,
				t1 = (l2ex * dy - l2ey * dx) / cross,
				t2 = (l1ex * dy - l1ey * dx) / cross;
			if(0 <= t1 && t1 <= 1 && 0 <= t2 && t2 <= 1) {
				var x = l1sx + t1 * l1ex,
					y = l1sy + t1 * l1ey;
				return [[x, y, {
					v1: v1,
					v2: v2,
					t1: Curve.getParameterOf(v1, x, y),
					t2: Curve.getParameterOf(v2, x, y)
				}]];
			}
		}
		return null;
	},
	getCurveLineIntersections: function(v1, v2) {
		var flip = Curve.isLinear(v1),
			vc = flip ? v2 : v1,
			vl = flip ? v1 : v2,
			lx1 = vl[0],
			ly1 = vl[1],
			lx2 = vl[6],
			ly2 = vl[7],
			ldx = lx2 - lx1,
			ldy = ly2 - ly1,
			angle = Math.atan2(-ldy, ldx),
			sin = Math.sin(angle),
			cos = Math.cos(angle),
			rlx2 = ldx * cos - ldy * sin,
			rvl = [0, 0, 0, 0, rlx2, 0, rlx2, 0],
			rvc = [];
		for(var i = 0; i < 8; i += 2) {
			var x = vc[i] - lx1,
				y = vc[i + 1] - ly1;
			rvc.push(
				x * cos - y * sin,
				y * cos + x * sin);
		}
		var roots = [],
			count = solveCubic(rvc, 1, 0, roots, 0, 1),
			ixs = [];
		for(var i = 0; i < count; i++) {
			var tc = roots[i],
				x = Curve.evaluate(rvc, tc)[0];
			if(x >= 0 && x <= rlx2) {
				var tl = Curve.getParameterOf(rvl, x, 0),
					t1 = flip ? tl : tc,
					t2 = flip ? tc : tl;
				ixs.push(Curve.evaluate(v1, t1).concat({
					v1: v1,
					v2: v2,
					t1: t1,
					t2: t2
				}));
				//Curve.evaluate(v2, t2)
			}
		}
		return ixs;
	},
	getCurveIntersections: function(v1, v2, range1, range2, recursion) {
		recursion = (recursion || 0) + 1;
		if(recursion > 20) {
			return [];
		}
		range1 = range1 || [0, 1];
		range2 = range2 || [0, 1];
		var part1		= Curve.getPartial(v1, range1[0], range1[1]),
			part2		= Curve.getPartial(v2, range2[0], range2[1]),
			iteration	= 0,
			ixs			= [];
		while(iteration++ < 20) {
			var range,
				intersects1 = clipFatLine(part1, part2, range = range2.slice()),
				intersects2 = 0;
			if(intersects1 === 0)
				break;
			if(intersects1 > 0) {
				range2 = range;
				part2 = Curve.getPartial(v2, range2[0], range2[1]);
				intersects2 = clipFatLine(part2, part1, range = range1.slice());
				if(intersects2 === 0)
					break;
				if(intersects1 > 0) {
					range1 = range;
					part1 = Curve.getPartial(v1, range1[0], range1[1]);
				}
			}
			if(intersects1 < 0 || intersects2 < 0) {
				if(range1[1] - range1[0] > range2[1] - range2[0]) {
					var t = (range1[0] + range1[1]) / 2;
					ixs = ixs.concat(Curve.getCurveIntersections(v1, v2, [range1[0], t], range2, recursion));
					ixs = ixs.concat(Curve.getCurveIntersections(v1, v2, [t, range1[1]], range2, recursion));
					break;
				} else {
					var t = (range2[0] + range2[1]) / 2;
					ixs = ixs.concat(Curve.getCurveIntersections(v1, v2, range1, [range2[0], t], recursion));
					ixs = ixs.concat(Curve.getCurveIntersections(v1, v2, range1, [t, range2[1]], recursion));
					break;
				}
			}
			if(Math.abs(range1[1] - range1[0]) <= 0.00001 &&
				Math.abs(range2[1] - range2[0]) <= 0.00001) {
				var t1 = (range1[0] + range1[1]) / 2,
					t2 = (range2[0] + range2[1]) / 2;
				ixs.push(Curve.evaluate(v1, t1).concat({
					v1: v1,
					v2: v2,
					t1: t1,
					t2: t2
				}));
				//Curve.evaluate(v2, t2)
				break;
			}
		}
		return ixs;
	},
	subdivide: function(v, t) {
		t = t || 0.5;
		var p1x = v[0],
			p1y = v[1],
			c1x = v[2],
			c1y = v[3],
			c2x = v[4],
			c2y = v[5],
			p2x = v[6],
			p2y = v[7];
		var u = 1 - t,
			p3x = u * p1x + t * c1x,
			p3y = u * p1y + t * c1y,
			p4x = u * c1x + t * c2x,
			p4y = u * c1y + t * c2y,
			p5x = u * c2x + t * p2x,
			p5y = u * c2y + t * p2y,
			p6x = u * p3x + t * p4x,
			p6y = u * p3y + t * p4y,
			p7x = u * p4x + t * p5x,
			p7y = u * p4y + t * p5y,
			p8x = u * p6x + t * p7x,
			p8y = u * p6y + t * p7y;
		return [
			[p1x, p1y, p3x, p3y, p6x, p6y, p8x, p8y], [p8x, p8y, p7x, p7y, p5x, p5y, p2x, p2y]
		];
	},
	getPartial: function(v, from, to) {
		if(from > 0)
			v = Curve.subdivide(v, from)[1];
		if(to < 1)
			v = Curve.subdivide(v, (to - from) / (1 - from))[0];
		return v;
	},
	getArea: function(v) {
		var p1x = v[0], p1y = v[1],
		c1x = v[2], c1y = v[3],
		c2x = v[4], c2y = v[5],
		p2x = v[6], p2y = v[7];
		return (  3.0 * c1y * p1x - 1.5 * c1y * c2x
				- 1.5 * c1y * p2x - 3.0 * p1y * c1x
				- 1.5 * p1y * c2x - 0.5 * p1y * p2x
				+ 1.5 * c2y * p1x + 1.5 * c2y * c1x
				- 3.0 * c2y * p2x + 0.5 * p2y * p1x
				+ 1.5 * p2y * c1x + 3.0 * p2y * c2x) / 10;
	},
	getNearestPoint: function(v, x, y) {
		var count		= 100,
			tolerance	= 10e-6,
			minDist		= Infinity,
			minT = 0;

		function refine(t) {
			if (t >= 0 && t <= 1) {
				var dist= distance2(Curve.evaluate(v, t), x, y);
					if (dist < minDist) {
						minDist = dist;
						minT = t;
						return true;
					}
			}
		}

		for (var i = 0; i <= count; i++) {
			refine(i / count);
		}

		var step = 1 / (count * 2);
		while (step > tolerance) {
			if (!refine(minT - step) && !refine(minT + step)) {
				step /= 2;
			}
		}
		var pt	= Curve.evaluate(v, minT),
			d	= Math.sqrt(distance2(pt, x, y));
		return {
			t:			minT,
			point:		pt,
			distance:	d
		};
		//return new CurveLocation(this, minT, pt, null, null, null, point.getDistance(pt));
	},
	
	getLength: function(v, from, to) {
		from	= from || 0;
		to		= to || 1;
		if ((v[0] - v[2])<=10e-12 && (v[1] - v[3])<=10e-12
				&& (v[6] - v[4])<=10e-12 && (v[7] - v[5])<=10e-12) {
			var dx = v[6] - v[0], 
				dy = v[7] - v[1]; 
			return (to - from) * Math.sqrt(dx * dx + dy * dy);
		}
		//getLenghtIntegrand
		var p1x = v[0], p1y = v[1],
			c1x = v[2], c1y = v[3],
			c2x = v[4], c2y = v[5],
			p2x = v[6], p2y = v[7],

			ax = 9 * (c1x - c2x) + 3 * (p2x - p1x),
			bx = 6 * (p1x + c2x) - 12 * c1x,
			cx = 3 * (c1x - p1x),

			ay = 9 * (c1y - c2y) + 3 * (p2y - p1y),
			by = 6 * (p1y + c2y) - 12 * c1y,
			cy = 3 * (c1y - p1y),

			//integrate
			n =  Math.max(2, Math.min(16, Math.ceil(Math.abs(to - from) * 32))),
			f = function(t) {
				var dx2 = (ax * t + bx) * t + cx,
					dy2 = (ay * t + by) * t + cy;
				return Math.sqrt(dx2 * dx2 + dy2 * dy2);
			},
			x = abscissas[n - 2],
			w = weights[n - 2],
			A = 0.5 * (to - from),
			B = A + from,
			i = 0,
			m = (n + 1) >> 1,
			sum = n & 1 ? w[i++] * f(B) : 0; 
		while (i < m) {
			var Ax = A * x[i];
			sum += w[i++] * (f(B + Ax) + f(B - Ax));
		}
		return A * sum;
	}
},
Curves = {
	getNearestPoint: function(vs, x, y) {
		var minDist = Infinity,
			minLoc = null;
		for (var i = 0, l = vs.length; i < l; i++) {
			var loc = Curve.getNearestPoint(vs[i], x, y);
			if (loc.distance < minDist) {
				minDist = loc.distance;
				minLoc = loc;
			}
		}
		return minLoc;
	},
	getIntersections: function(vs1, vs2) {
		var ixs = [];
		for(var i = 0; i < vs1.length; i++) {
			var l1 = Curve.isLinear(vs1[i]);
			for(var j = 0; j < vs2.length; j++) {
				var is = null,
					l2 = Curve.isLinear(vs2[j]);
				//in- and out-handles same as previous endpoint?
				if(l1 && l2) {
					is = Curve.getLineIntersection(vs1[i], vs2[j]);
				} else if(l1 || l2) {
					is = Curve.getCurveLineIntersections(vs1[i], vs2[j]);
				} else {
					is = Curve.getCurveIntersections(vs1[i], vs2[j]);
				}
				if(is && is.length) {
					ixs = ixs.concat(is);
				}
			}
		}
		return ixs;
	},
	getLength: function(vs) {
		var length = 0;
		for(var i = 0; i<vs.length; i++) {
			length += Curve.getLength(vs[i]);
		}
		return length;
	}
},
Path = {
	//vs = absolute Path Array
	transformAbsolutePathArray: function(vs, matrix, dec) {

		// For arc parameter rounding
		var arc_dec = (dec !== false) ? 6 : false,
			r = function(num) {
				if(dec !== false) return Math.round(num * Math.pow(10, dec)) / Math.pow(10, dec);
				else return num;
			},
			ra = function(num) {
				if(arc_dec !== false) return Math.round(num * Math.pow(10, arc_dec)) / Math.pow(10, arc_dec);
				else return num;
			};

		// Rounding coordinates to dec decimals
		if(dec || dec === 0) {
			if(dec > 15) dec = 15;
			else if(dec < 0) dec = 0;
		} else {
			dec = false;
		}

		arc_dec = (dec && dec > 6) ? dec : arc_dec;

		// Get the relation matrix that converts path coordinates
		// to SVGroot's coordinate space

		// The following code can bake transformations
		// both normalized and non-normalized data
		// Coordinates have to be Absolute in the following
		var i = 0,
			j, m = vs.length,
			letter = "",
			x = 0,
			y = 0,
			point, newcoords = [],
			pt = {},
			subpath_start = null;
		for(; i < m; i++) {
			letter = vs[i][0].toUpperCase();
			newcoords[i] = [];
			newcoords[i][0] = vs[i][0];

			if(letter == "A") {
				x = vs[i][6];
				y = vs[i][7];

				pt.x = vs[i][6];
				pt.y = vs[i][7];
				newcoords[i] = SVG.utils.arcTransform(vs[i][1], vs[i][2], vs[i][3], vs[i][4], vs[i][5], pt, matrix);
				// rounding arc parameters
				// x,y are rounded normally
				// other parameters at least to 5 decimals
				// because they affect more than x,y rounding
				newcoords[i][1] = ra(newcoords[i][1]); //rx
				newcoords[i][2] = ra(newcoords[i][2]); //ry
				newcoords[i][3] = ra(newcoords[i][3]); //x-axis-rotation
				newcoords[i][6] = r(newcoords[i][6]); //x
				newcoords[i][7] = r(newcoords[i][7]); //y
			} else if(letter != "Z") {
				// parse other segs than Z and A
				for(j = 1; j < vs[i].length; j = j + 2) {
					if(letter == "V") {
						y = vs[i][j];
					} else if(letter == "H") {
						x = vs[i][j];
					} else {
						x = vs[i][j];
						y = vs[i][j + 1];
					}
					pt.x = x;
					pt.y = y;
					SVG.utils.matrixTransform(pt, matrix);
					newcoords[i][j] = r(pt.x);
					newcoords[i][j + 1] = r(pt.y);
				}
			}
			if((letter != "Z" && subpath_start === null) || letter == "M") {
				subpath_start = {
					x: x,
					y: y
				};
			}
			if(letter == "Z" && subpath_start) {
				x = subpath_start.x;
				y = subpath_start.y;
			}
			if(letter == "V" || letter == "H") {
				newcoords[i][0] = "L";
			}
		}
		return newcoords;
	},
	toAbsolute: function(vs) {
		var res = [],
			x = 0,
			y = 0,
			mx = 0,
			my = 0,
			start = 0;
		if(vs[0][0].toUpperCase() === "M") {
			x = vs[0][1];
			y = vs[0][2];
			mx = x;
			my = y;
			start++;
			vs[0] = res[0] = ["M", x, y];
		}
		var crz = vs.length === 3 && vs[0][0] == "M" && vs[1][0].toUpperCase() == "R" && vs[2][0].toUpperCase() == "Z";
		for(var i = start; i < vs.length; i++) {
			var r = [],
				vi = vs[i];
			cmd = vi[0].toUpperCase();
			if(vi[0] !== cmd) {
				r.push(cmd);
				switch(cmd) {
					case "A":
						r[1] = vi[1];
						r[2] = vi[2];
						r[3] = vi[3];
						r[4] = vi[4];
						r[5] = vi[5];
						r[6] = vi[6]+x;
						r[7] = vi[7]+y;
						break;
					case "V":
						r.push(vi[1] + y);
						break;
					case "H":
						r.push(vi[1] + x);
						break;
					case "R":
						var dots = [x, y].concat(vi.slice(1));
						for(var j = 2; j < dots.length; j++) {
							dots[j] += x;
							dots[++j] += y;
						}
						res.pop();
						res = res.concat(catmullRom2bezier(dots, crz));
						break;
					case "M":
						mx = vi[1] + x;
						my = vi[2] + y;
					default:
						for(var j = 1; j < vi.length; j++) {
							r[j] = vi[j] + ((j % 2) ? x : y);
						}
				}
			} else if(vi[0] === "R") {
				dots = [x, y].concat(vi.slice(1));
				res.pop();
				res = res.concat(catmullRom2bezier(dots, crz));
				r = ["R"].concat(vi.slice(-2));
			} else {
				for(var j = 0; j < vi.length; j++) {
					r[j] = vi[j];
				}
			}
			switch(r[0]) {
			case "Z":
				x = mx;
				y = my;
				break;
			case "H":
				x = r[1];
				break;
			case "V":
				y = r[1];
				break;
			case "M":
				mx = r[r.length - 2];
				my = r[r.length - 1];
			default:
				x = r[r.length - 2];
				y = r[r.length - 1];
			}

			res.push(r);
		}
		return res;
	},
	toRelative: function(vs) {
		var res = [],
			x = 0,
			y = 0,
			mx = 0,
			my = 0,
			start = 0;
		if(vs[0][0] == "M") {
			x = vs[0][1];
			y = vs[0][2];
			mx = x;
			my = y;
			start++;
			res.push(["M", x, y]);
		}
		for(var i = start; i < vs.length; i++) {
			var r = res[i] = [],
				vi = vs[i],
				len;
			if(vi[0] != vi[0].toLowerCase()) {
				r[0] = vi[0].toLowerCase();
				switch(r[0]) {
				case "a":
					r[1] = vi[1];
					r[2] = vi[2];
					r[3] = vi[3];
					r[4] = vi[4];
					r[5] = vi[5];
					r[6] = (vi[6] - x).toFixed(3);
					r[7] = (vi[7] - y).toFixed(3);
					break;
				case "v":
					r[1] = (vi[1] - y)
						.toFixed(3);
					break;
				case "m":
					mx = vi[1];
					my = vi[2];
				default:
					for(var j = 1; j < vi.length; j++) {
						r[j] = (vi[j] - ((j % 2) ? x : y))
							.toFixed(3);
					}
				}
			} else {
				if(vi[0] == "m") {
					mx = vi[1] + x;
					my = vi[2] + y;
				}
				for(var k = 0, kk = vi.length; k < kk; k++) {
					res[i][k] = vi[k];
				}
			}
			len = res[i].length;
			switch(res[i][0]) {
			case "z":
				x = mx;
				y = my;
				break;
			case "h":
				x += +res[i][len - 1];
				break;
			case "v":
				y += +res[i][len - 1];
				break;
			default:
				x += +res[i][len - 2];
				y += +res[i][len - 1];
			}
		}
		return res;
	},
	//Thanks to Timo for fixing Raphael.js
	//http://jsbin.com/oqojan/53
	//Thanks to Dmitry Baranovskiy for Raphael.js / path2curve()
	toCurve: (function() {
		var a2c = function(x1, y1, rx, ry, angle, largeArcFlag, sweepFlag, x2, y2, recursive) {
			// for more information of where this math came from visit:
			// http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
			var _120 = Math.PI * 120 / 180,
				rad = Math.PI / 180 * (angle || 0),
				res = [],
				f1, f2, cx, cy;
			if(!recursive) {
				var xy = rotate(x1, y1, -rad),
					cos = Math.cos(Math.PI / 180 * angle),
					sin = Math.sin(Math.PI / 180 * angle),
					x1 = xy.x;
				y1 = xy.y;
				xy = rotate(x2, y2, -rad);
				x2 = xy.x;
				y2 = xy.y;
				var x = (x1 - x2) / 2,
					y = (y1 - y2) / 2,
					h = (x * x) / (rx * rx) + (y * y) / (ry * ry);
				if(h > 1) {
					h = Math.sqrt(h);
					rx = h * rx;
					ry = h * ry;
				}
				var rx2 = rx * rx,
					ry2 = ry * ry,
					k = (largeArcFlag == sweepFlag ? -1 : 1) *
						Math.sqrt(Math.abs((rx2 * ry2 - rx2 * y * y - ry2 * x * x) / (rx2 * y * y + ry2 * x * x)));
				cx = k * rx * y / ry + (x1 + x2) / 2,
				cy = k * -ry * x / rx + (y1 + y2) / 2,
				f1 = Math.asin(((y1 - cy) / ry)
					.toFixed(9)),
				f2 = Math.asin(((y2 - cy) / ry)
					.toFixed(9));

				f1 = x1 < cx ? Math.PI - f1 : f1;
				f2 = x2 < cx ? Math.PI - f2 : f2;
				f1 < 0 && (f1 = Math.PI * 2 + f1);
				f2 < 0 && (f2 = Math.PI * 2 + f2);
				if(sweepFlag && f1 > f2) {
					f1 = f1 - Math.PI * 2;
				}
				if(!sweepFlag && f2 > f1) {
					f2 = f2 - Math.PI * 2;
				}
			} else {
				f1 = recursive[0];
				f2 = recursive[1];
				cx = recursive[2];
				cy = recursive[3];
			}
			var df = f2 - f1;
			if(Math.abs(df) > _120) {
				var f2old = f2,
					x2old = x2,
					y2old = y2;
				f2 = f1 + _120 * (sweepFlag && f2 > f1 ? 1 : -1);
				x2 = cx + rx * Math.cos(f2);
				y2 = cy + ry * Math.sin(f2);
				res = a2c(x2, y2, rx, ry, angle, 0, sweepFlag, x2old, y2old, [f2, f2old, cx, cy]);
			}
			df = f2 - f1;
			var c1 = Math.cos(f1),
				s1 = Math.sin(f1),
				c2 = Math.cos(f2),
				s2 = Math.sin(f2),
				t = Math.tan(df / 4),
				hx = 4 / 3 * rx * t,
				hy = 4 / 3 * ry * t,
				m1 = [x1, y1],
				m2 = [x1 + hx * s1, y1 - hy * c1],
				m3 = [x2 + hx * s2, y2 - hy * c2],
				m4 = [x2, y2];
			m2[0] = 2 * m1[0] - m2[0];
			m2[1] = 2 * m1[1] - m2[1];
			if(recursive) {
				return [m2, m3, m4].concat(res);
			} else {
				res = [m2, m3, m4].concat(res)
					.join()
					.split(",");
				var newres = [];
				for(var i = 0; i < res.length; i++) {
					newres[i] = i % 2 ? rotate(res[i - 1], res[i], rad)
						.y : rotate(res[i], res[i + 1], rad)
						.x;
				}
				return newres;
			}
		},
		q2c = function(x1, y1, ax, ay, x2, y2) {
			var _13 = 1 / 3,
				_23 = 2 / 3;
			return [
				_13 * x1 + _23 * ax,
				_13 * y1 + _23 * ay,
				_13 * x2 + _23 * ax,
				_13 * y2 + _23 * ay,
				x2,
				y2
			];
		},
		l2c = function(x1, y1, x2, y2) {
			return [x1, y1, x2, y2, x2, y2];
		};
		return function(vs, raw) {
			vs	= Path.toAbsolute(vs);
			if(!vs.length) {
				return (raw?[0, 0]:["C"]).concat([0, 0, 0, 0, 0, 0]);
			}
			var curves	= [],
				attrs	= {
					x: 0,
					y: 0,
					bx: 0,
					by: 0,
					X: 0,
					Y: 0,
					qx: null,
					qy: null
				},
				cmd, prevCmd;
			//for(var i = 0; i < vs.length; i++) {
			while(v = vs.shift()) {
				cmd = v[0];
				var curve;

				if(cmd !== 'T' && cmd !== 'Q') {
					attrs.qx = attrs.qy = null;
				}

				pre = raw === true?[attrs.x, attrs.y] : ["C"];

				switch(cmd) {
					case "M":
						attrs.X = v[1];
						attrs.Y = v[2];
						pre = raw === true?pre:[cmd];
						curve = v.slice(1);
						break;
					case "A":
						curve = a2c.apply(0, [attrs.x, attrs.y].concat(v.slice(1)));
						break;
					case "S":
						var nx, ny;
						if(prevCmd == "C" || prevCmd == "S") { // previous cmd C/S ?
							nx = attrs.x * 2 - attrs.bx;
							ny = attrs.y * 2 - attrs.by;
						} else { // otherwise
							nx = attrs.x;
							ny = attrs.y;
						}
						curve = [nx, ny].concat(v.slice(1));
						break;
					case "T":
						if(prevCmd === "Q" || prevCmd === "T") { // previous cmd Q/T ?
							attrs.qx = attrs.x * 2 - attrs.qx;
							attrs.qy = attrs.y * 2 - attrs.qy;
						} else { //otherwise
							attrs.qx = attrs.x;
							attrs.qy = attrs.y;
						}
						curve = q2c(attrs.x, attrs.y, attrs.qx, attrs.qy, v[1], v[2]);
						break;
					case "Q":
						attrs.qx = v[1];
						attrs.qy = v[2];
						curve = q2c(attrs.x, attrs.y, v[1], v[2], v[3], v[4]);
						break;
					case "L":
						curve = l2c(attrs.x, attrs.y, v[1], v[2]);
						break;
					case "H":
						curve = l2c(attrs.x, attrs.y, v[1], attrs.y);
						break;
					case "V":
						curve = l2c(attrs.x, attrs.y, attrs.x, v[1]);
						break;
					case "Z":
						curve = l2c(attrs.x, attrs.y, attrs.X, attrs.Y);
						break;
					default:
						curve = v.slice(1);
				}

				var grabCurve = function() {
					var c = curve.splice(0,6);
					if(!(raw && cmd === 'M')) {
						curves.push(pre.concat(c));
					}
					var length = c.length;
					attrs.x		= c[length - 2];
					attrs.y		= c[length - 1];
					attrs.bx	= parseFloat(c[length - 4]) || attrs.x;
					attrs.by	= parseFloat(c[length - 3]) || attrs.y;
					pre = raw === true?[attrs.x, attrs.y] : ["C"];
					return curve.length;
				}

				prevCmd = cmd;
				var hasMore = grabCurve();
				while(hasMore) {
					prevCmd = 'A';
					hasMore = grabCurve();
				}

			}

			return curves;
		};
	})(),
	fromCurve: function(vs) {
		if(!vs.length) {
			return [];
		}
		var path = [["M", vs[0][0], vs[0][1]]];
		for(var i = 0; i<vs.length; i++) {
			path.push(["C"].concat(vs[i].slice(2)));
		}
		return path;
	}
}

SVG.utils = {
	arcTransform:                   arcTransform,
	matrixTransform:                matrixTransform,
	isLinear:                       Curve.isLinear,
	subdivideCurve:                 Curve.subdivide,
	getCurveBounds:                 Curve.getBounds,
	getLength:                      Curves.getLength,
	getNearestPoint:                Curves.getNearestPoint,
	getIntersections:               Curves.getIntersections,
	transformAbsolutePathArray:     Path.transformAbsolutePathArray,
	toRelative:                     Path.toRelative,
	toAbsolute:                     Path.toAbsolute,
	toCurve:                        Path.toCurve,
	fromCurve:                      Path.fromCurve
};
