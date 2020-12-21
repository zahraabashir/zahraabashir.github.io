---
title: 'Advanced RL&DL Deepmind: Introduction to Machine Learning Based AI'
date: 2020-12-21
permalink: /posts/2020/12/advanced-rl-and-dl-sess1/
tags:
  - reinforcement learning
  - deep learning
---

Introduction
------
Creating the Artificial General Intelligence undoubtly is one of the most important problems that mankid has faced and also could transform our world greatly once again. I think the wonder of this could be expressed precisely by the Deepmind's slogan:
<strong>What if solving one problem could unlock solutions to thousands more?</strong>
Moreover, there is a belief that Deep Reinforcement Learning could be foundation for Artificial General Intelligence, so maybe because of that there is an incredible upward trend twords Reinforcement Learning throughout past years, see this tweet from chris manning!

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The amazing rise of reinforcement learning!<br>(With graph neural networks and meta-learning in hot pursuit. ConvNets? Tired.) Based on <a href="https://twitter.com/hashtag/ICLR2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICLR2021</a> keywords HT <a href="https://twitter.com/PetarV_93?ref_src=twsrc%5Etfw">@PetarV_93</a> <a href="https://t.co/ozKpNUVH1i">pic.twitter.com/ozKpNUVH1i</a></p>&mdash; Christopher Manning (@chrmanning) <a href="https://twitter.com/chrmanning/status/1332725903470706688?ref_src=twsrc%5Etfw">November 28, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

---
Deepmind is one of the most prominent companies (if not the most!) working on this field. Deepmind's [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) was a great breakthrough in the history of this field. AlphaGo succeeded to defeat a professional go player a decade before expected! Moreover, Deepmind has released a lot of great courses on different aspects of artifical intelligence literature, one of them is this course on RL and DL tought by Thore Graepel and some other great guest lecturers.
So, in this series of blog posts, I want to share with you what I learnt from [Deepmind](https://deepmind.com/) & UCL Advanced Deep Learning & Reinforcement Learning Course.

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