use lyon::{math::point, path::PathEvent};
use swash::{
    scale::outline::Outline,
    zeno::{Command, PathData},
};

pub(super) fn collect_events(outline: &Outline, events: &mut Vec<PathEvent>) {
    events.clear();
    let mut first = point(0.0, 0.0);
    let mut last = first;
    for command in outline.path().commands() {
        let event = match command {
            Command::MoveTo(to) => {
                first = point(to.x, -to.y);
                last = first;
                PathEvent::Begin { at: first }
            }
            Command::LineTo(to) => {
                let from = last;
                last = point(to.x, -to.y);
                PathEvent::Line { from, to: last }
            }
            Command::QuadTo(control, to) => {
                let from = last;
                last = point(to.x, -to.y);
                PathEvent::Quadratic {
                    from,
                    ctrl: point(control.x, -control.y),
                    to: last,
                }
            }
            Command::CurveTo(control1, control2, to) => {
                let from = last;
                last = point(to.x, -to.y);
                PathEvent::Cubic {
                    from,
                    ctrl1: point(control1.x, -control1.y),
                    ctrl2: point(control2.x, -control2.y),
                    to: last,
                }
            }
            Command::Close => PathEvent::End {
                last,
                first,
                close: true,
            },
        };
        events.push(event);
    }
}
