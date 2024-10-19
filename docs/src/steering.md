# Steering

```admonish warning
It's tempting to instead define
`nudge = sae.decoder(onehot)`,
where `onehot = torch.tensor([0,...,0,1,0,...,0])` is a unit vector pointing in the direction
of the feature we want to amplify.

The problem is that `sae.decoder` multiplies by the weight matrix
(picking out the correct feature vector) but _also_ adds the bias.
This bias is large enough to mess up steering, i.e.
adding in this corrupted `nudge` doesn't make the model generate
text that resembles the desired feature.
```


strength 10 gets the feature to appear at least half the time
although at the cost of some model consistency

I tuned feature strength to 10, and then generated two steered examples (limited to 100 tokens)
for each of features 0, 1, 2, 6, without cherrypicking

Feature 0

> Once upon a time, there was a little boy named Tim. He went on vacation with his mom, who put on a big bed. in small jail, so she said thank you. Tim was in jail, but in her while she was. They had to get in their room and make a big, clear room. Tim was so happy to enter the room and make everything in.

After the door,, Tim was in a room with many passages. She,thanks for her home

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ boy ▁ named ▁ Tim ▁. ▁ He ▁ went ▁ on ▁ vacation ▁ with ▁ his ▁ mom ▁, ▁ who ▁ put ▁ on ▁ a ▁ big ▁ bed ▁. ▁ in ▁ small ▁ jail ▁, ▁ so ▁ she ▁ said ▁ thank ▁ you ▁. ▁ Tim ▁ was ▁ in ▁ jail ▁, ▁ but ▁ in ▁ her ▁ while ▁ she ▁ was ▁. ▁ They ▁ had ▁ to ▁ get ▁ in ▁ their ▁ room ▁ and ▁ make ▁ a ▁ big ▁, ▁ clear ▁ room ▁. ▁ Tim ▁ was ▁ so ▁ happy ▁ to ▁ enter █ the ▁ room ▁ and ▁ make ▁ everything ▁ in ▁. ▁
 ▁
 ▁After ▁ the ▁ door ▁, ▁, ▁ Tim ▁ was ▁ in ▁ a ▁ room ▁ with ▁ many ▁ passages ▁. ▁ She ▁, ▁thanks ▁ for ▁ her ▁ home ▁ 
 
 
>  Once upon a time, there was a little girl named Lily. She had a silver crown and her brother had won before.. She was a good boy, who wincedog means so smooth in."

Lily and her brother were both so rich. They had, and the big brother made her smile. Even though she had to leave, like in jail, rich to get in with Ben.

They both had surprise, and had to say goodbye, so they got

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ girl ▁ named ▁ Lily ▁. ▁ She ▁ had ▁ a ▁ silver ▁ crown ▁ and ▁ her ▁ brother ▁ had ▁ won ▁ before ▁. ▁. ▁ She ▁ was ▁ a ▁ good ▁ boy ▁, ▁ who ▁ win █ced ▁og ▁ means ▁ so ▁ smooth ▁ in ▁." ▁  ▁
 ▁
 ▁L ▁ily ▁ and ▁ her ▁ brother ▁ were ▁ both ▁ so ▁ rich ▁. ▁ They ▁ had ▁, ▁ and ▁ the ▁ big ▁ brother ▁ made ▁ her ▁ smile ▁. ▁ Even ▁ though ▁ she ▁ had ▁ to ▁ leave ▁, ▁ like ▁ in ▁ jail ▁, ▁ rich ▁ to ▁ get ▁ in ▁ with ▁ Ben ▁. ▁  ▁
 ▁
 ▁They ▁ both ▁ had ▁ surprise ▁, ▁ and ▁ had ▁ to ▁ say ▁ goodbye ▁, ▁ so ▁ they ▁ got ▁

Feature 1


Steered output:
Once upon a time, there was a little bird, and each day.

One day, when it was a bit of a wind, when it saw a cloud, a small bird.

Then, a lot of more. Then, it learned something new.

Every day and every morning, and every night, until it had two weeks of.
 many months and months.

Finally, when it stopped, but mom and dad.

Then, many more.

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ little ▁ bird ▁, ▁ and ▁ each ▁ day ▁. ▁
 ▁
 ▁One ▁ day ▁, ▁ when ▁ it ▁ was ▁ a ▁ bit ▁ of ▁ a ▁ wind ▁, ▁ when ▁ it ▁ saw ▁ a ▁ cloud ▁, ▂ a ▅ small ▆ bird ▁. ▁
 ▁
 ▁Then ▁, ▁ a ▁ lot ▁ of ▁ more ▁. ▁ Then ▁, ▁ it ▁ learned ▁ something ▁ new ▁. ▁
 ▁
 ▁Every ▁ day ▁ and ▂ every ▅ morning ▁, ▁ and ▁ every █ night ▁, ▂ until ▁ it ▁ had ▁ two ▁ weeks ▁ of ▁. ▁
 ▁ many ▁ months ▁ and ▁ months ▁. ▁
 ▁
 ▁Finally ▁, ▁ when ▁ it ▁ stopped ▁, ▁ but ▁ mom ▁ and ▁ dad ▁. ▁
 ▁
 ▁Then ▁, ▁ many ▁ more ▁. ▁
 
 
 
 > Once upon a time, there was a wise old man who liked to travel. He liked to eat, he was very curious and he wanted a new, sweet sometimes. He loved to learn new things.

Every time he visited a new place, he wanted to watch television, or guess some more. He sometimes played a game or a pretend game. Then he guessed a story, and made some more games.

Eventually, it was very fun.

"Are so many

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ wise ▁ old ▁ man ▁ who ▁ liked ▁ to ▁ travel ▁. ▁ He ▁ liked ▁ to ▁ eat ▁, ▁ he ▁ was ▁ very ▁ curious ▁ and ▁ he ▁ wanted ▁ a ▁ new ▁, ▁ sweet ▁ sometimes ▁. ▁ He ▁ loved ▁ to ▁ learn ▁ new ▁ things ▁. ▁  ▁
 ▁
 ▁Every ▁ time ▁ he ▁ visited ▁ a ▁ new ▁ place ▁, ▁ he ▁ wanted ▁ to ▁ watch ▁ television ▁, ▁ or ▁ guess ▁ some ▁ more ▁. ▁ He ▁ sometimes ▁ played ▁ a ▁ game ▁ or ▂ a █ pretend ▂ game ▄. ▁ Then ▁ he ▁ guessed ▁ a ▁ story ▁, ▂ and ▁ made ▁ some ▁ more ▁ games ▁. ▁
 ▁
 ▁Eventually ▁, ▁ it ▁ was ▁ very ▁ fun ▁. ▁
 ▁
 ▁" ▁Are ▁ so ▁ many ▁


Feature 2


Once upon a time, there was a group to play with to a group for a long time to for the one to go to to the nation for a parade. "Time to go for the parade to for the parade to for the last to go for the Wheel. to the Wheel to the room for the Wheel. to remove the group for the tank to to exit for the to to exit.

As for the for the group to go, the excited for to be for the new exit

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ group ▁ to ▁ play ▁ with ▁ to ▁ a ▁ group ▁ for ▁ a ▁ long ▁ time ▁ to ▁ for ▁ the ▁ one ▁ to ▁ go ▁ to ▁ to ▁ the ▁ nation ▁ for ▁ a ▁ parade ▁. ▁ " ▁Time █ to ▁ go ▁ for ▁ the ▁ parade ▁ to ▁ for ▁ the ▁ parade ▁ to ▁ for ▁ the ▁ last ▁ to ▁ go ▁ for ▁ the ▁ Wheel ▁. ▁ to ▁ the ▁ Wheel ▁ to ▁ the ▁ room ▁ for ▁ the ▁ Wheel ▁. ▁ to ▁ remove ▁ the ▁ group ▁ for ▁ the ▁ tank ▁ to ▁ to ▁ exit ▁ for ▁ the ▁ to ▁ to ▁ exit ▁. ▁
 ▁
 ▁As ▁ for ▁ the ▁ for ▁ the ▁ group ▁ to ▁ go ▁, ▁ the ▁ excited ▁ for ▁ to ▁ be ▁ for ▁ the ▁ new ▁ exit ▁
 
 
 Once upon a time, there was a family for dinner. They all sat down for dinner for the meal to eat.

Before dinner for dinner, for dinner to be for dinner for everyone to eat. Mom to order the family to eat to eat dinner for dinner.

After dinner, for a full meal for dinner for the family to eat for dinner. for dinner for dessert their meal for them to be for dinner every year where they were for the dessert.

The day to

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ family ▁ for ▁ dinner ▁. ▁ They ▁ all ▁ sat ▁ down ▁ for ▁ dinner ▁ for ▁ the ▁ meal ▁ to ▁ eat ▁. ▁
 ▁
 ▁Before ▁ dinner ▁ for ▁ dinner ▁, ▁ for ▁ dinner ▁ to ▁ be ▁ for ▁ dinner ▁ for ▁ everyone ▁ to ▁ eat ▁. ▁ Mom ▁ to ▁ order ▁ the ▁ family ▁ to ▁ eat ▁ to ▁ eat ▁ dinner ▁ for ▁ dinner ▁. ▁
 ▁
 ▁After ▁ dinner ▁, ▁ for ▁ a ▁ full ▁ meal ▁ for ▁ dinner ▁ for ▁ the ▁ family ▁ to ▁ eat ▁ for ▁ dinner ▁. ▁ for ▁ dinner ▁ for ▁ dessert ▁ their ▁ meal ▁ for ▁ them ▁ to ▁ be ▁ for ▁ dinner ▁ every ▁ year ▁ where ▁ they ▁ were ▁ for ▁ the ▁ dessert ▁. ▁
 ▁
 ▁The ▁ day ▁ to ▁
 


Feature 6


> Once upon a time, there was an ignorant driver. One day, the cars that was traveling quickly, and so did a terrible crash. The lights went so quick
ly, that was very loud.
<|endoftext|>

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ an ▁ ignorant ▁ driver ▁. ▁ One ▁ day ▁, ▁ the ▁ cars █ that ▁ was ▁ traveling ▁ quickly ▁, ▁ and ▁ so ▁
did ▁ a ▁ terrible ▁ crash ▇. ▁ The ▁ lights ▁ went ▅ so ▁ quickly ▁, ▁ that ▁ was ▁ very ▁ loud ▂. ▁
 ▁<|endoftext|> ▁
 
 
 
 Once upon a time, there was a train. On one of the roads, a very big crash happened. It made a lot of noise. "". crash the cars and truck.," and cars," crash with a loud noise. That noise and accident.

One day, red and noisy cars came to distant. The river, which is really a rough and dangerous. Jark.

The people in the traffic and noise and traffic crash, ever.

The red

Now feed the steered text into an unmodified LLM, and print how much the SparseAutoEncoder thinks the LLM activates on the feature:
Once ▁ upon ▁ a ▁ time ▁, ▁ there ▁ was ▁ a ▁ train ▂. ▁ On ▁ one ▁ of ▁ the ▁ roads ▁, ▁ a ▁ very ▁ big ▁ crash ▆ happened ▁. ▁ It ▁ made ▁ a ▁ lot ▁ of ▁ noise ▂. ▁ " ▁". ▁ crash ▆ the ▁ cars ▄ and ▁ truck ▁. ▁," ▁ and ▁ cars ▃," ▁ crash ▄ with ▁ a ▁ loud ▁ noise ▂. ▁ That ▁ noise ▅ and ▁ accident ▃. ▁
 ▁
 ▁One ▁ day ▁, ▁ red ▁ and ▁ noisy ▂ cars ▃ came ▁ to ▁ distant ▁. ▁ The ▁ river ▅, ▁ which ▂ is ▁ really ▁ a ▁ rough ▁ and ▁ dangerous ▂. ▁  ▁J ▂ark ▁. ▁
 ▁
 ▁The ▁ people ▁ in ▁ the ▁ traffic ▅ and ▁ noise ▄ and ▁ traffic ▆ crash █, ▁ ever ▁. ▁  ▁
 ▁
 ▁The ▁ red ▁
 
 
 
 
 

in the top 10 examples from the TinyStories validation set, this feature only activates on "fire" (or more weakly on "firemen").

The fire neuron really seems to be a danger neuron,
based on the steering results

Look at long tail to confirm this
Indeed, we see some non-fire related examples:
> " ▁Mom ▁my ▁, ▁ what ▁'s ▁ happening ▁?\" ▁ she ▁ cried ▁. ▁ \" ▁It ▁'s ▁ an ▁ earthquake ▆, ▁ Lily ▁,\" ▁ her ▁ mom ▁my ▁ said ▁
and
> Tim ▁my ▁'s ▁ mom ▁my ▁ told ▁ him ▁ to ▁ come ▁ inside ▁ because ▁ there ▁ might ▁ be ▁ a ▁ flood ▇. ▁ Tim ▁my ▁ didn ▁'t ▁ know ▁ what ▁ a ▁ flood █ was ▁, ▁ but ▁ he ▁ listened ▁ to ▁ his ▁ mom ▁my ▁ and ▁ went ▁ inside ▁. ▁
