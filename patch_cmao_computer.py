import re

with open("src/cmao/cmao.py", "r") as f:
    text = f.read()

# Make the total advantage normalized from total reward, matching GRPO
new_compute = """    def compute_group(self, group: ScoredGroup) -> ScoredGroup:
        correctness = [1.0 if item.score.answer_correct else 0.0 for item in group.scored_samples]
        quality = [item.score.quality_score for item in group.scored_samples]
        
        mode_bonus = [0.0 for _ in group.scored_samples]
        correct_indices = [idx for idx, value in enumerate(correctness) if value > 0.0]
        if correct_indices:
            counts = Counter(group.scored_samples[idx].score.mode_label for idx in correct_indices)
            total = len(correct_indices)
            for idx in correct_indices:
                mode = group.scored_samples[idx].score.mode_label
                mode_probability = counts[mode] / total
                mode_bonus[idx] = quality[idx] * (-math.log(mode_probability))

        # Standard GRPO: combine into total reward first, THEN normalize!
        total_rewards = []
        for idx in range(len(group.scored_samples)):
            r = self.lambda_ans * correctness[idx] + self.lambda_qual * quality[idx] + self.lambda_mode * mode_bonus[idx]
            total_rewards.append(r)
            
        mean_r = _mean(total_rewards)
        std_r = _std(total_rewards, self.epsilon)
        
        updated_samples: list[ScoredSample] = []
        for idx, item in enumerate(group.scored_samples):
            a_total = (total_rewards[idx] - mean_r) / std_r if std_r > 0 else 0.0
            updated_samples.append(
                ScoredSample(
                    sample=item.sample,
                    score=item.score,
                    advantage=AdvantageBundle(
                        a_ans=correctness[idx],
                        a_qual=quality[idx],
                        a_mode=mode_bonus[idx],
                        a_total=a_total,
                    ),
                )
            )
        return ScoredGroup(problem=group.problem, scored_samples=updated_samples, metadata=group.metadata)"""

# Replace the compute_group function
text = re.sub(r'    def compute_group\(self, group: ScoredGroup\) -> ScoredGroup:.*?(?=\n\n|$)', new_compute, text, flags=re.DOTALL)

with open("src/cmao/cmao.py", "w") as f:
    f.write(text)

