use crate::core::UnsignedPhysicalRect;
use rstar::{ParentNode, RTree, RTreeNode, RTreeObject, AABB};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::ControlFlow;

#[derive(Clone, Copy)]
enum DamageEdge {
    Left,
    Right,
    Top,
    Bottom,
}

impl DamageEdge {
    const ALL: [Self; 4] = [Self::Left, Self::Right, Self::Top, Self::Bottom];

    fn growth_coordinate(self, bounds: UnsignedPhysicalRect) -> u32 {
        match self {
            Self::Left => u32::MAX - bounds.min.x,
            Self::Right => bounds.max.x,
            Self::Top => u32::MAX - bounds.min.y,
            Self::Bottom => bounds.max.y,
        }
    }

    fn blocking(envelope: AABB<[f64; 2]>, bounds: UnsignedPhysicalRect) -> Option<(Self, u32)> {
        let lower = envelope.lower();
        let upper = envelope.upper();
        let thresholds = [
            u32::MAX - upper[0] as u32,
            lower[0] as u32,
            u32::MAX - upper[1] as u32,
            lower[1] as u32,
        ];
        Self::ALL
            .into_iter()
            .zip(thresholds)
            .find(|&(edge, threshold)| threshold >= edge.growth_coordinate(bounds))
    }
}

struct NodeLocation {
    parent: usize,
    child: usize,
}

/// Resumes blocked spatial branches only when a growing damage edge reaches them.
#[derive(Default)]
pub(super) struct DamageTraversal {
    node_locations: Vec<NodeLocation>,
    path: Vec<usize>,
    waiting_nodes: [BinaryHeap<Reverse<(u32, usize)>>; 4],
}

impl DamageTraversal {
    pub(super) fn clear(&mut self) {
        self.node_locations.clear();
        self.path.clear();
        for waiting in &mut self.waiting_nodes {
            waiting.clear();
        }
    }

    fn ready_node(&mut self, bounds: UnsignedPhysicalRect) -> Option<usize> {
        for edge in DamageEdge::ALL {
            let waiting = &mut self.waiting_nodes[edge as usize];
            if waiting
                .peek()
                .is_some_and(|&Reverse((threshold, _))| threshold < edge.growth_coordinate(bounds))
            {
                return waiting.pop().map(|Reverse((_, location))| location);
            }
        }
        None
    }

    fn node_at<'tree, T>(
        &mut self,
        root: &'tree ParentNode<T>,
        mut location: usize,
    ) -> &'tree RTreeNode<T>
    where
        T: RTreeObject,
    {
        self.path.clear();
        while location != 0 {
            let node = &self.node_locations[location];
            self.path.push(node.child);
            location = node.parent;
        }
        let mut children = root.children();
        for &child in self.path[1..].iter().rev() {
            let RTreeNode::Parent(parent) = &children[child] else {
                unreachable!("only spatial parent nodes have children");
            };
            children = parent.children();
        }
        &children[self.path[0]]
    }

    fn visit_children<T>(
        &mut self,
        parent: &ParentNode<T>,
        location: usize,
        viewport: UnsignedPhysicalRect,
        bounds: &mut UnsignedPhysicalRect,
        expand: &mut impl FnMut(&T, &mut UnsignedPhysicalRect),
    ) -> ControlFlow<()>
    where
        T: RTreeObject<Envelope = AABB<[f64; 2]>>,
    {
        for (child, node) in parent.children().iter().enumerate() {
            let child_location = self.node_locations.len();
            self.node_locations.push(NodeLocation {
                parent: location,
                child,
            });
            self.visit_node(node, child_location, viewport, bounds, expand)?;
        }
        ControlFlow::Continue(())
    }

    fn visit_node<T>(
        &mut self,
        node: &RTreeNode<T>,
        location: usize,
        viewport: UnsignedPhysicalRect,
        bounds: &mut UnsignedPhysicalRect,
        expand: &mut impl FnMut(&T, &mut UnsignedPhysicalRect),
    ) -> ControlFlow<()>
    where
        T: RTreeObject<Envelope = AABB<[f64; 2]>>,
    {
        if let Some((edge, threshold)) = DamageEdge::blocking(node.envelope(), *bounds) {
            self.waiting_nodes[edge as usize].push(Reverse((threshold, location)));
            return ControlFlow::Continue(());
        }
        match node {
            RTreeNode::Parent(parent) => {
                self.visit_children(parent, location, viewport, bounds, expand)
            }
            RTreeNode::Leaf(trigger) => {
                expand(trigger, bounds);
                if *bounds == viewport {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            }
        }
    }

    /// Each branch can wait once per edge. Heap operations and path lookup cost O(log n).
    pub(super) fn expand<T>(
        &mut self,
        tree: &RTree<T>,
        viewport: UnsignedPhysicalRect,
        mut bounds: UnsignedPhysicalRect,
        mut expand: impl FnMut(&T, &mut UnsignedPhysicalRect),
    ) -> UnsignedPhysicalRect
    where
        T: RTreeObject<Envelope = AABB<[f64; 2]>>,
    {
        self.clear();
        // Location zero represents the root and has no parent.
        self.node_locations.push(NodeLocation {
            parent: 0,
            child: 0,
        });
        if self
            .visit_children(tree.root(), 0, viewport, &mut bounds, &mut expand)
            .is_break()
        {
            return bounds;
        }
        while let Some(location) = self.ready_node(bounds) {
            let node = self.node_at(tree.root(), location);
            if self
                .visit_node(node, location, viewport, &mut bounds, &mut expand)
                .is_break()
            {
                break;
            }
        }
        bounds
    }
}

#[cfg(test)]
mod tests;
