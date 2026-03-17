class ExperimentConfig:
    def __init__(self, experiment, actor, transform, terrain, live_recording=False, video=False, contact_forces=False, total_frames=10000000, lr=3e-4, max_grad_norm=1,
                 frames_per_batch=2048, sub_batch_size=64, num_epochs=10,
                 clip_epsilon=0.2, gamma=0.99, lmbda=0.95, entropy_eps=1e-4):
        self.experiment = experiment
        self.actor = actor
        self.transform = transform
        self.terrain = terrain
        self.video = video
        self.contact_forces = contact_forces
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.file_path = f"Logs/{experiment}"
        self.live_recording = live_recording


mlp_config = ExperimentConfig(
    experiment = "mlp_long_v2",
    actor = "mlp_actor",
    transform = "Notransform",
    terrain = "flat",
    entropy_eps=0.02,      # was 1e-4
    num_epochs=4,           # was 10
    sub_batch_size=256,     # was 64
    #total_frames = 5_000_000
    )

joint_config = ExperimentConfig(
    experiment="joint_long_v2",
    actor="joint_actor",
    transform="joint_centipede_graph",
    terrain="flat",
    entropy_eps=0.02,      # was 1e-4
    num_epochs=4,           # was 10
    sub_batch_size=256,     # was 64
)

leg_config = ExperimentConfig(
    experiment="leg_long_v2",
    transform="leg_centipede_graph",
    actor="leg_actor",
    terrain='flat',
    entropy_eps=0.02,      # was 1e-4
    num_epochs=4,           # was 10
    sub_batch_size=256,     # was 64
)
