use std::f64::EPSILON;
use std::sync::Arc;

use nalgebra::Vector2;

use crate::geometry::Point;
use crate::global::*;
use crate::quad_tree::node::*;

pub struct Body {
    /// An Arc (atomic reference counting) smart pointer to a QuadNode
    node: Arc<QuadNode>,
    pub position: Point,
    pub velocity: Vector2<f64>,
    pub acceleration: Vector2<f64>,
}

impl Body {
    /// Sets the node field to a new QuadNode that is "ready" for insertion into the quadtree, using the current position field.
    pub fn make_ready(&mut self) {
        self.node = make_ready(self.position.clone(), self.node.clone())
    }

    /// Performs collision detection using the position and node fields, and updates the velocity field accordingly.
    pub fn collision_detect(&mut self) {
        let impact = collision_detect(&self.position, self.node.clone());
        self.velocity.x += impact.x;
        self.velocity.y += impact.y;
    }

    /// Calculates the gravitational impact on the body based on its position field and the root quadtree, and updates the acceleration field accordingly.
    pub fn gravity_impact(&mut self, root: Arc<QuadNode>) {
        let impact = get_impact(&self.position, root);
        self.acceleration.x = impact.0 / self.position.mass;
        self.acceleration.y = impact.1 / self.position.mass;
    }

    /// Updates the position field based on the velocity field and a constant ALPHA.
    pub fn update_position(&mut self) {
        self.position.x += self.velocity.x * ALPHA;
        self.position.y += self.velocity.y * ALPHA;
    }

    /// v = a * t
    pub fn update_velocity(&mut self) {
        self.velocity.x += self.acceleration.x * ALPHA;
        self.velocity.y += self.acceleration.y * ALPHA;
    }

    /// Converts the position field to an sdl2::rect::Point object, which is used for rendering in SDL2.
    pub fn geometric(&self) -> sdl2::rect::Point {
        sdl2::rect::Point::new(self.position.x as i32, self.position.y as i32)
    }

    /// Constructs a new Body object with the given x, y, mass, and root quadtree.
    pub fn new(x: f64, y: f64, mass: f64, root: Arc<QuadNode>) -> Body {
        let position = Point { x, y, mass };
        Body {
            node: insert(root, position.clone()),
            position,
            velocity: Vector2::new(0.0, 0.0),
            acceleration: Vector2::new(0.0, 0.0),
        }
    }

    /// Reinserts the Body object into the quadtree, updating the node field based on the current position field.
    pub fn reinsert(&mut self, root: Arc<QuadNode>) {
        self.node = insert(root, self.position.clone());
    }

    /// Checks if the Body object has crossed the boundaries of the simulation window, and if so, updates its position and velocity fields accordingly.
    pub fn check_boundary(&mut self) {
        let real_width = *WIDTH / *SCALE_FACTOR;
        let real_height = *HEIGHT / *SCALE_FACTOR;
        if self.position.x + RADIUS >= real_width {
            self.position.x = real_width - RADIUS - EPSILON;
            self.velocity.x = -self.velocity.x * 0.5;
        }
        if self.position.x - RADIUS <= 0.0 {
            self.position.x = RADIUS + EPSILON;
            self.velocity.x = -self.velocity.x * 0.5;
        }
        if self.position.y + RADIUS >= real_height {
            self.position.y = real_height - RADIUS - EPSILON;
            self.velocity.y = -self.velocity.y * 0.5;
        }
        if self.position.y - RADIUS <= 0.0 {
            self.position.y = RADIUS + EPSILON;
            self.velocity.y = -self.velocity.y * 0.5;
        }
    }
}

/// SimpleBody has fields for x, y, m (mass), vx (velocity x), vy (velocity y), ax (acceleration x), and ay (acceleration y). It has a single method to_sdl() which converts the x and y fields to an sdl2::rect::Point object, which is used for rendering in SDL2.
pub struct SimpleBody {
    pub x: f64,
    pub y: f64,
    pub m: f64,
    pub vx: f64,
    pub vy: f64,
    pub ax: f64,
    pub ay: f64,
}

impl SimpleBody {
    pub fn to_sdl(&self) -> sdl2::rect::Point {
        sdl2::rect::Point::new(self.x as i32, self.y as i32)
    }
}
