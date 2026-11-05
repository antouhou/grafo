use crate::commands::{DrawClip, RenderCommand, RenderOperation, TextureComposite};
use smallvec::{IntoIter, SmallVec};

// Entering a node can copy its backdrop, begin a target, composite its mask,
// capture and filter its backdrop, and draw the shape.
const INLINE_COMMAND_CAPACITY: usize = 6;

pub(super) type PendingCommands<'scene> =
    IntoIter<[RenderCommand<'scene>; INLINE_COMMAND_CAPACITY]>;

/// Commands emitted by one traversal step. A step enters or leaves at most one node.
#[derive(Default)]
pub(crate) struct TraversalCommands<'scene> {
    commands: SmallVec<[RenderCommand<'scene>; INLINE_COMMAND_CAPACITY]>,
}

impl<'scene> TraversalCommands<'scene> {
    pub(super) fn push(&mut self, operation: RenderOperation<'scene>) {
        self.push_command(RenderCommand {
            operation,
            clip: DrawClip::default(),
        });
    }

    pub(super) fn push_composite(&mut self, composite: TextureComposite, clip: DrawClip) {
        self.push_command(RenderCommand {
            operation: RenderOperation::CompositeTexture(composite),
            clip,
        });
    }

    pub(super) fn push_command(&mut self, command: RenderCommand<'scene>) {
        self.commands.push(command);
    }
}

impl<'scene> IntoIterator for TraversalCommands<'scene> {
    type Item = RenderCommand<'scene>;
    type IntoIter = PendingCommands<'scene>;

    fn into_iter(self) -> Self::IntoIter {
        self.commands.into_iter()
    }
}
