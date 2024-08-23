from types import SimpleNamespace

# all tunable Parameters
config = SimpleNamespace(**{})

config.filename = "what-are-three-words-that-describe-you-201475.mp3"

config.temperature = 0.2

config.sampling_rate = 16000

config.output_speed = 1

config.output_pitch = 1

config.output_voice = "Male"


if __name__ == "__main__":
    print(config)