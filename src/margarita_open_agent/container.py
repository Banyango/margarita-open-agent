import wireup

from margarita_open_agent import core, libs

container = wireup.create_async_container(
    injectables=[core, libs],
)
