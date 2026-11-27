use winit::keyboard::KeyCode;

#[derive(Clone, Copy)]
pub enum Action {
    Start,
    Stop,
    ToggleRedrawOverlay,
    Exit,
}

pub struct Button {
    pub label: &'static str,
    pub bounds: [(f32, f32); 2],
    pub action: Action,
    shortcut: KeyCode,
}

impl Button {
    const fn new(
        x: f32,
        width: f32,
        label: &'static str,
        shortcut: KeyCode,
        action: Action,
    ) -> Self {
        Self {
            label,
            bounds: [(x, 40.0), (x + width, 76.0)],
            action,
            shortcut,
        }
    }
}

pub const BUTTONS: [Button; 3] = [
    Button::new(408.0, 80.0, "Start", KeyCode::Enter, Action::Start),
    Button::new(496.0, 80.0, "Stop", KeyCode::Space, Action::Stop),
    Button::new(
        584.0,
        192.0,
        "Highlight screen updates",
        KeyCode::KeyD,
        Action::ToggleRedrawOverlay,
    ),
];

pub fn key_action(code: KeyCode) -> Option<Action> {
    BUTTONS
        .iter()
        .find(|button| button.shortcut == code)
        .map(|button| button.action)
        .or(match code {
            KeyCode::Escape | KeyCode::BrowserBack => Some(Action::Exit),
            _ => None,
        })
}

pub fn pointer_action(position: [f32; 2]) -> Option<Action> {
    BUTTONS
        .iter()
        .find(|button| {
            let [(left, top), (right, bottom)] = button.bounds;
            (left..right).contains(&position[0]) && (top..=bottom).contains(&position[1])
        })
        .map(|button| button.action)
}
