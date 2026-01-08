# BaseExt.py

class base_ext:
    """
    Minimal reusable base for TouchDesigner extensions.
    Provides GetPage() to get-or-create a custom parameter page.
    """
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp

    def Log(self, *args):
        print(f"[{self.ownerComp.path}]", *args)

    def GetPage(self, pageName, create_if_missing=True):
        """
        Return a custom Page on self.ownerComp with the given name.
        If it doesn't exist and create_if_missing=True, create it.

        Matching is case-insensitive against page.name and page.label.
        Returns the Page, or None if not found and create_if_missing=False.
        """
        if not pageName:
            raise ValueError("GetPage: pageName must be a non-empty string.")

        lname = pageName.lower()
        for p in self.ownerComp.customPages:
            label = getattr(p, 'label', p.name)
            if p.name.lower() == lname or str(label).lower() == lname:
                return p

        if create_if_missing:
            return self.ownerComp.appendCustomPage(pageName)

        return None
