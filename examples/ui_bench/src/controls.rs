use winit::keyboard::KeyCode;

pub const BUTTON_SIZE: [f32; 2] = [80.0, 36.0];

#[derive(Clone, Copy)]
pub enum Action {
    Start,
    Stop,
    Exit,
}

pub struct Button {
    pub label: &'static str,
    pub bounds: [(f32, f32); 2],
    pub action: Action,
    shortcut: KeyCode,
}

impl Button {
    const fn new(index: usize, label: &'static str, shortcut: KeyCode, action: Action) -> Self {
        let x = 408.0 + index as f32 * 88.0;
        Self {
            label,
            bounds: [(x, 40.0), (x + BUTTON_SIZE[0], 40.0 + BUTTON_SIZE[1])],
            action,
            shortcut,
        }
    }
}

pub const BUTTONS: [Button; 2] = [
    Button::new(0, "Start", KeyCode::Enter, Action::Start),
    Button::new(1, "Stop", KeyCode::Space, Action::Stop),
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
