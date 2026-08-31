"""Ce que l'adaptation à l'écoute doit faire — et surtout ce qu'elle ne doit pas.

Les cas négatifs comptent plus que les positifs : une conversion manquée
s'entend une fois, une conversion fautive fait dire à l'auteur le contraire de
ce qu'il a écrit, dans quarante-six livres à la fois.
"""
import pytest

from narration.adresse_audio import adapter, rapport_texte


def adapte(texte: str) -> str:
    return adapter(texte)[0]


class TestIntertitres:
    def test_le_titre_le_plus_frequent(self):
        # Présent dans dix-neuf livres sur quarante-six.
        assert adapte("Comment lire ce livre") == "Comment écouter ce livre audio"

    def test_le_titre_garde_sa_suite(self):
        assert adapte("Comment lire ce livre, et comment l'utiliser") == \
            "Comment écouter ce livre audio, et comment l'utiliser"

    def test_un_chapitre_ne_devient_pas_audio(self):
        # « audio » se dit du livre, une fois ; le répéter par chapitre lasse.
        assert adapte("Comment lire ce chapitre") == "Comment écouter ce chapitre"

    def test_le_titre_doit_etre_seul_sur_sa_ligne(self):
        # Au milieu d'un paragraphe, c'est une phrase, pas un intertitre :
        # la règle des verbes s'en charge, sans ajouter « audio ».
        assert "ce livre audio" not in adapte("Il explique comment lire ce livre en trois jours.")


class TestVerbes:
    @pytest.mark.parametrize("avant, apres", [
        ("Vous lisez ce livre par curiosité.", "Vous écoutez ce livre par curiosité."),
        ("Si vous avez lu ce livre jusqu'ici.", "Si vous avez écouté ce livre jusqu'ici."),
        ("Vous pouvez lire ce livre comme une exploration.",
         "Vous pouvez écouter ce livre comme une exploration."),
        ("En lisant ce livre, vous comprendrez.", "En écoutant ce livre, vous comprendrez."),
        ("Les parents qui lisent ce chapitre le savent.",
         "Les parents qui écoutent ce chapitre le savent."),
        ("Ceux qui liront ces pages y trouveront un appui.",
         "Ceux qui écouteront ces pages y trouveront un appui."),
        ("Relisez ce chapitre demain.", "Réécoutez ce chapitre demain."),
    ])
    def test_conjugaisons(self, avant, apres):
        assert adapte(avant) == apres

    def test_la_casse_est_conservee(self):
        assert adapte("Lisez ce livre lentement.") == "Écoutez ce livre lentement."

    def test_reference_avant_le_verbe(self):
        assert adapte("Ce livre se lit d'une traite.") == "Ce livre s'écoute d'une traite."

    @pytest.mark.parametrize("avant, apres", [
        # Tous les remplacements commencent par une voyelle, ce que « lire »
        # ne faisait pas : le mot d'avant doit s'élider.
        ("Ce livre se lit vite.", "Ce livre s'écoute vite."),
        ("Ce chapitre ne se lit pas seul.", "Ce chapitre ne s'écoute pas seul."),
        ("Je lis ce livre le soir.", "J'écoute ce livre le soir."),
    ])
    def test_elision(self, avant, apres):
        assert adapte(avant) == apres

    def test_la_distance_est_bornee(self):
        # Sans borne, un « lire » de la page d'avant s'accrocherait à un
        # « ce livre » de la page d'après.
        loin = "lire " + "x" * 80 + " ce livre"
        assert adapte(loin) == loin


class TestCeQuiNeDoitPasBouger:
    def test_la_lecture_en_general(self):
        t = "Apprendre à lire transforme le cerveau."
        assert adapte(t) == t

    def test_un_autre_livre(self):
        # Mesuré dans livre-06 : il s'agit d'un manuel de réseau cité.
        t = "Des ingénieurs se sont formées en lisant cet ouvrage de référence."
        assert adapte(t) == t

    @pytest.mark.parametrize("t", [
        # Chacun relevé dans le corpus, dans sa phrase.
        "Des applications non compatibles avec les lecteurs d'écran.",
        "Payer par simple contact de la main avec un lecteur.",
        "Elle a consacré sa carrière à l'étude du cerveau lecteur.",
        "Lui-même, lecteur assidu depuis l'enfance, peine à finir un chapitre.",
        "Les lecteurs numériques avaient intériorisé un mode de balayage.",
        "Des plateformes qui relient annonceurs et lecteurs pour les journaux.",
        "Quand le récit implique un auditeur ou un lecteur, la relation compte.",
        "Comme d'autres se présenteraient comme lecteurs de Proust.",
    ])
    def test_lecteur_qui_ne_designe_pas_le_public(self, t):
        assert adapte(t) == t

    def test_lire_sans_reference_au_livre(self):
        t = "Elle a lu trois romans cet été."
        assert adapte(t) == t


class TestLecteurDevientAuditeur:
    @pytest.mark.parametrize("avant, apres", [
        ("À vous, lecteurs, qui avez la curiosité.", "À vous, auditeurs, qui avez la curiosité."),
        ("Le lecteur est invité à expérimenter.", "L'auditeur est invité à expérimenter."),
        ("Ce livre s'adresse à plusieurs catégories de lecteurs.",
         "Ce livre s'adresse à plusieurs catégories d'auditeurs."),
        ("Chacune offre de l'inspiration pour les lectrices contemporaines.",
         "Chacune offre de l'inspiration pour les auditrices contemporaines."),
    ])
    def test_le_public_devient_auditeur(self, avant, apres):
        assert adapte(avant) == apres

    def test_un_livre_peut_etre_exclu_en_entier(self):
        # « L'EFFET PODCAST » oppose page après page la lecture et l'écoute.
        t = "Le format du livre demande à son lecteur ce que le podcast demande à son auditeur."
        assert adapter(t, slug="livre-21-podcast")[0] == t
        # Le même texte, dans un autre livre, se convertit.
        assert "auditeur" in adapter("Le lecteur trouvera ici des pistes.")[0]


class TestSignalements:
    def test_lecteur_ecarte_est_signale(self):
        _, _, sig = adapter("L'étude du cerveau lecteur est ancienne.")
        assert [s.motif for s in sig] == ["lecteur"]

    def test_le_support_bloque_la_conversion(self):
        # Mesuré dans livre-01. « écouter ce livre sur une liseuse » serait
        # pire que la phrase d'origine : la conversion doit être refusée.
        texte = "Vous allez peut-être lire ce livre sur une liseuse ou un téléphone."
        neuf, ch, sig = adapter(texte)
        assert neuf == texte
        assert not ch
        assert any(s.motif == "support" for s in sig)

    def test_un_crayon_ne_bloque_pas(self):
        # Écrire en écoutant est possible : « crayon » n'est pas un support
        # de lecture, il ne doit pas empêcher une conversion juste.
        neuf, ch, _ = adapter("Écoutez-moi : lisez ce livre, crayon à la main.")
        assert "écoutez ce livre" in neuf.lower()
        assert ch

    def test_cet_ouvrage_est_signale_sans_etre_touche(self):
        texte = "Ils se sont formés en lisant cet ouvrage."
        neuf, ch, sig = adapter(texte)
        assert neuf == texte
        assert not ch
        assert any(s.motif == "cet ouvrage" for s in sig)

    def test_le_rapport_dit_les_deux(self):
        _, ch, sig = adapter("Comment lire ce livre\nL'étude du cerveau lecteur.")
        r = rapport_texte(ch, sig)
        assert "1 adaptation(s)" in r
        assert "1 passage(s)" in r
        assert "cerveau lecteur" in r


class TestTexteEntier:
    def test_un_passage_reel(self):
        # Extrait de livre-01, tel qu'il est dans la file.
        source = (
            "Comment lire ce livre\n\n"
            "Ce livre a été conçu pour s'adapter à un cerveau qui a perdu "
            "l'habitude de la lecture longue. Vous lisez ce livre par curiosité "
            "plus que par nécessité."
        )
        neuf, ch, _ = adapter(source)
        assert neuf.startswith("Comment écouter ce livre audio")
        assert "Vous écoutez ce livre" in neuf
        # « l'habitude de la lecture longue » parle de lecture en général.
        assert "lecture longue" in neuf
        assert len(ch) == 2
