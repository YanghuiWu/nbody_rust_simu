use std::fmt::{Debug, Error, Formatter};
use std::hash::{Hash, Hasher};

use nalgebra::Vector2;
// use num::Float;

use crate::global::RADIUS;

/// Square represents a rectangle with sides parallel to the x and y axes, defined by its two opposite corners, self.0 and self.1, both of type Vector2<f64> from the nalgebra library
#[derive(Copy, Clone)]
pub struct Square(pub Vector2<f64>, pub Vector2<f64>);

impl Debug for Square {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "{:?}",
            [((self.0).x, (self.0).y), ((self.1).x, (self.1).y)]
        )
    }
}

impl Square {
    /// checks whether a given Point is completely contained within the Square.
    pub(crate) fn contains(&self, x: &Point) -> bool {
        self.0.x > x.x + RADIUS
            && self.0.y > x.y + RADIUS
            && self.1.x < x.x - RADIUS
            && self.1.y < x.y - RADIUS
    }

    /// checks whether a given Point is within a distance of RADIUS from the Square.
    // pub fn touch(&self, x: &Point) -> bool {
    //     static DIST: f64 = RADIUS * RADIUS;
    //     (self.0.x - x.x) * (self.0.x - x.x) <= DIST
    //         || (self.1.x - x.x) * (self.1.x - x.x) <= DIST
    //         || (self.0.y - x.y) * (self.0.y - x.y) <= DIST
    //         || (self.1.y - x.y) * (self.1.y - x.y) <= DIST
    // }

    /// checks whether a given Point is within a distance of RADIUS from the Square.
    // pub fn can_touch(&self, x: &Point) -> bool {
    //     static DIST: f64 = 9.0 * RADIUS * RADIUS;
    //     let mid = (self.0 + self.1) / 2.0;
    //     (mid.x - x.x) * (mid.x - x.x) <= DIST
    //         || (mid.y - x.y) * (mid.y - x.y) <= DIST
    // }

    // Rewrite previous two methods Replaced touch method in Square with more efficient implementations that use vector operations and take advantage of Rust's SIMD features.

    /// checks whether a given Point is within a distance of RADIUS from the Square.
    pub fn touch(&self, point: &Point) -> bool {
        let dist = RADIUS * RADIUS;
        let p = point.coords();
        let d = |i: usize| (p[i] - self.0[i]).max(self.1[i] - p[i]).max(0.0);
        let dx = d(0);
        let dy = d(1);
        dx * dx + dy * dy <= dist
    }

    /// checks whether a given Point is within a distance of 3 * RADIUS from the Square.
    pub fn can_touch(&self, point: &Point) -> bool {
        let dist = 9.0 * RADIUS * RADIUS;
        let mid = (self.0 + self.1) / 2.0;
        let dx = point.x - mid.x;
        let dy = point.y - mid.y;
        dx * dx + dy * dy <= dist
    }
}

/// Point represents a point in two-dimensional space, defined by its x and y coordinates and a mass.
#[derive(PartialEq, Copy, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub mass: f64,
}

impl Debug for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", (self.x, self.y))
    }
}

/// Point implements the Eq trait, which allows it to be compared for equality with other Points.
impl Eq for Point {}

/// Point implements the Hash trait, which allows it to be used as a key in a HashMap. ??? Not sure
impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // (self.x.integer_decode(), self.y.integer_decode(), self.mass.integer_decode()).hash(state);
        (self.x.to_bits(), self.y.to_bits(), self.mass.to_bits()).hash(state);
        // Changed the hashing function for Point to use to_bits instead of integer_decode, which is simpler and more efficient.
    }
}

impl Point {
    /// return a new Vector2<f64> containing the point's coordinates
    pub fn coords(&self) -> Vector2<f64> {
        Vector2::new(self.x, self.y)
    }

    /// return a new Point with the same coordinates as the given Vector2<f64>
    pub fn update_by(&mut self, data: &Vector2<f64>) {
        self.x += data.x;
        self.y += data.y;
    }
}

/// check whether two Points are within a distance of 2 * RADIUS from each other.
pub fn check(p: &Point, q: &Point) -> bool {
    let a = p.x - q.x;
    let b = p.y - q.y;
    a * a + b * b < 4.0 * RADIUS * RADIUS
}
