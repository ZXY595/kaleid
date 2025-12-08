use odometries::algorithm::lio::{self, ImuInit, ImuMeasured, StampedImu};

fn main() {
    let interval = 0.2;
    let mut fake_imus = std::iter::repeat_with(|| rand::random_range(-0.01..0.01))
        .map(|x| ImuMeasured::new(0., 0., 9.81 + x, 0. + x, 0. + x, 0. + x))
        .enumerate()
        .map(|(t, measured)| StampedImu::new(t as f64 * interval, measured));

    let imu_init = fake_imus
        .by_ref()
        .take(20)
        .collect::<Option<ImuInit<_>>>()
        .unwrap();

    let mut lio = imu_init.new_lio(lio::Config::default());

    let fake_imus = fake_imus.take(
        option_env!("TIMES")
            .and_then(|s| {
                s.parse()
                    .inspect_err(|e| eprintln!("Invalid TIMES: {e}"))
                    .ok()
            })
            .unwrap_or(1000),
    );
    lio.extend(fake_imus);

    let pose = lio.get_pose().translation;
    println!("{:?}", pose);
}
