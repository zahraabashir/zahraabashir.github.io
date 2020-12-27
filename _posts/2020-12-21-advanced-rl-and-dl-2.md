---
title: 'Advanced RL&DL Course Deepmind and UCL'
date: 2020-12-21
permalink: /posts/2020/12/advanced-rl-and-dl-2/
tags:
  - reinforcement learning
  - deep learning
---

# Deep Learning 1: Introduction to Machine Learning Based AI

The first session is on introducing the lecturers and a definition of intelligence.

What is intelligence?
======
Maybe at the first place it seems a very vague and broad question but actually there is an answer for it from this [paper](https://doi.org/10.1007/s11023-007-9079-x):
<center>$$ \underset{Measure\ of\ Intelligence}{\Upsilon(\pi)} := \underset{Sum\ over\ environments}{\Large{\underset{\mu \in E}{\Sigma}}} \underset{Complexity\ penalty}{2^{-K(\mu)}} *  \underset{Value\ achivied}{V_{\mu}^{\pi}} $$</center>

It describe that the inteligence of a being(agent) is determind by the amount of value(reward) that it can get from different environments. $\pi$ is the policy of the agent in different environments; $k$ is the [kolmogorov complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity) of the specified environment; and $V$ is the reward that the agent would get on that environment. One of the natural question that arised is "which environment to sum over?", aforementioned paper sayed that this environment must be computable, which means it is required that envoronmental probability is computable. For example, rolling a dice and writing the results in a sequence is not computable, but the probability function that describe this experiment is computable, thusly, rolling a dice could one of us possible computable environment. Another counter-intuitive thing that may take our attention to itself in the formula is the negative sign behind the kolmogorov complextity. The reason of adding that may not seem very obvious at the first glimpse, but it has to be there as we have exponentialy more complex environment compare to simpler ones, so this negative sign imply that for an agent to be intelligent, it's better to first perform well on more simpler environment then as it goes furthur switch to more complex ones.

Computer Games
-------
As you may notice, computer games seem a perfect fit for this problems. Interestingly, there are other advantages, for instance we can access unlimited data by running the games in a relative less time in comparison to other machine learning fields.

the rest of this session is about the concpets of deep learning which are not our main focus for now.

the video of the first session
-------
<iframe width="600" height="400" src="https://www.youtube.com/embed/iOh7QUZGyiU?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>