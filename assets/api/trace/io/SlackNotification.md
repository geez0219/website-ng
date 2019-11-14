## SlackNotification
```python
SlackNotification(channel, end_msg, begin_msg=None, token=None, verbose=0)
```
Send message to Slack channel when training begins and ends.In order to send messages to Slack, user needs to generate a Slack token for authentication. Once a token isgenerated, assign it to the class initializer argument `token`, or to the environment variable `SLACK_TOKEN`.
* **For Slack token generation, see** : 
* **https** : //api.slack.com/custom-integrations/legacy-tokens

#### Args:

* **channel (str)** :  A string. Can be either channel name or user id.
* **end_msg (Union[str, function])** :  The message to send to the Slack channel when training starts, can be either a        string or a function. If this is a function, it can take the state dict as input.
* **begin_msg (str, optional)** :  The message to send to the Slack channel when training starts. Defaults to None.
* **token (str, optional)** :  This token can be generated from Slack API. Defaults to None. When the value is None,        this argument will read from the environment variable `SLACK_TOKEN`.

#### Raises:

* **TypeError** :  If `begin_msg` or `end_msg` is not (str, function).