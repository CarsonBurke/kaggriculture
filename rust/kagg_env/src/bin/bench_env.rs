use _kagg_env::{CompactAction, Game, GameConfig, PLAYERS};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let games = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);
    let mut states: Vec<Game> = (0..games)
        .map(|seed| Game::new(seed as u64, GameConfig::default()))
        .collect();
    let actions = [CompactAction::default(); PLAYERS];
    let started = Instant::now();
    states.par_iter_mut().for_each(|game| {
        while !game.done {
            black_box(game.step(black_box(&actions)));
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    let transitions = games * 719;
    println!(
        "games={games} seconds={elapsed:.6} games_per_second={:.1} joint_transitions_per_second={:.0}",
        games as f64 / elapsed,
        transitions as f64 / elapsed,
    );
}
